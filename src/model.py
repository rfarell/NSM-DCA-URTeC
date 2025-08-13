import torch
import torch.nn as nn
import torch.nn.functional as F

########################################################
# 1) ExpertGP: Single GP Expert with Full 3×3 Noise Model
########################################################
class ExpertGP(nn.Module):
    """
    A single expert GP using Random Fourier Features (RFF) with a global
    variational posterior over its weights (shared across all wells)
    and a full 3×3 observation noise covariance.
    """
    def __init__(self, state_dim=3, z_dim=19, num_basis=16, K=10):
        super().__init__()
        self.state_dim = state_dim
        self.z_dim = z_dim
        self.num_basis = num_basis
        self.K = K  # Number of posterior samples

        # RFF input dimension: [state_dim + 1 (for time) + z_dim]
        self.rff_input_dim = state_dim + 1 + z_dim

        # -----------------------------
        # Spectral Mixture Kernel Parameters
        # -----------------------------
        # Number of mixture components (Q in the manuscript)
        self.num_mixtures = 3  # Can be adjusted
        
        # Mixture weights (log-scale for positivity)
        self.log_mixture_weights = nn.Parameter(torch.zeros(self.num_mixtures))
        
        # Mixture means (frequencies) - one per component per input dimension
        # Initialize with diverse frequencies
        freq_init = torch.zeros(self.num_mixtures, self.rff_input_dim)
        for i in range(self.num_mixtures):
            # Initialize with different frequency scales
            freq_init[i] = torch.randn(self.rff_input_dim) * (0.1 + 0.5 * i)
        self.mixture_means = nn.Parameter(freq_init)
        
        # Mixture variances (lengthscales) - log-scale for positivity
        # Initialize with different scales
        var_init = torch.zeros(self.num_mixtures, self.rff_input_dim)
        for i in range(self.num_mixtures):
            var_init[i] = torch.log(torch.ones(self.rff_input_dim) * (0.5 + 0.5 * i))
        self.log_mixture_variances = nn.Parameter(var_init)

        # -----------------------------
        #  Global b and L for the RFF weights
        #  (single distribution, shared across wells)
        # -----------------------------
        nb_total = 2 * num_basis  # for sine and cosine
        tril_size = (num_basis * (num_basis + 1)) // 2

        self.global_b = nn.Parameter(torch.zeros(nb_total))
        self.global_L_flat = nn.Parameter(torch.zeros(tril_size))
        # -----------------------------

        # For spectral mixture, we sample frequencies dynamically
        # No fixed epsilon tensor needed

        # Unconstrained lower-triangular for 3×3 noise covariance
        tril_size_cov = 3 * (3 + 1) // 2  # = 6
        # self.noise_unconstrained = nn.Parameter(torch.zeros(tril_size_cov))
        self.noise_unconstrained = nn.Parameter(torch.randn(tril_size_cov) * 0.01)
        
        # Decay parameters for algebraic taper (one per phase)
        # Initialize log_tc to log(365) ~ 5.9 for more reasonable decline curves
        self.log_tc = nn.Parameter(torch.ones(self.state_dim) * 5.9)      # log days
        self.tilde_alpha = nn.Parameter(torch.zeros(self.state_dim)) # unconstrained
        
        # Cache tril indices - will be moved to correct device later
        self.register_buffer('_tril_idx_n', torch.tril_indices(num_basis, num_basis))
        self.register_buffer('_tril_idx_3', torch.tril_indices(3, 3))

    def compute_bL(self):
        """
        Returns the single global mean b and lower-triangular L for the RFF weights,
        shared by all wells.
        """
        # b: shape [2*num_basis]
        b = self.global_b
        # L_flat: shape [(num_basis*(num_basis+1))//2]
        L_flat = self.global_L_flat

        n = self.num_basis
        L = torch.zeros(n, n, device=b.device)
        L[self._tril_idx_n[0], self._tril_idx_n[1]] = L_flat
        # Ensure diagonal is positive
        L.diagonal().copy_(F.softplus(L.diagonal() + 1e-6))
        return b, L

    def sample_weights(self):
        """
        Draw K samples from the global posterior N(b, L L^T).
        Returns w_samples: [K, 2*num_basis].
        """
        b, L = self.compute_bL()  # shape [2*num_basis], [n, n]
        nb_total = 2 * self.num_basis

        # Build block-diagonal for sine and cosine expansions: shape [2n, 2n]
        sqrt_C = torch.block_diag(L, L)  # [2n, 2n]

        # Sample K times from N(b, sqrt_C) - vectorized
        eps = torch.randn(self.K, nb_total, device=b.device)
        w_samples = b + (sqrt_C @ eps.T).T  # [K, 2*num_basis]
        return w_samples

    def get_noise_cov(self):
        """
        Constructs the full 3×3 noise covariance Sigma.
        """
        device = self.noise_unconstrained.device
        L_noise = torch.zeros(3, 3, device=device)
        L_noise[self._tril_idx_3[0], self._tril_idx_3[1]] = self.noise_unconstrained
        # Ensure diagonal is positive
        L_noise.diagonal().copy_(F.softplus(L_noise.diagonal() + 1e-6))
        Sigma = L_noise @ L_noise.transpose(0, 1)
        # Add a small regularization to the diagonal for stability
        Sigma = Sigma + torch.eye(3, device=device) * 1e-6
        return Sigma
    
    def sample_spectral_frequencies(self, num_samples):
        """
        Sample frequencies from the spectral mixture distribution.
        Returns: [num_samples, rff_input_dim]
        """
        # Get mixture parameters
        weights = F.softmax(self.log_mixture_weights, dim=0)  # [num_mixtures]
        means = self.mixture_means  # [num_mixtures, rff_input_dim]
        variances = torch.exp(self.log_mixture_variances)  # [num_mixtures, rff_input_dim]
        
        # Sample mixture components
        component_samples = torch.multinomial(weights, num_samples, replacement=True)
        
        # Sample frequencies from selected components - vectorized
        frequencies = means[component_samples] + torch.sqrt(variances[component_samples]) * torch.randn(num_samples, self.rff_input_dim, device=means.device)
        
        return frequencies

    def _taper(self, t):
        """
        Compute (1 + t/t_c)^(-alpha) with alpha = 1+eps+softplus(tilde_alpha)
        t : tensor [..., 1]  (scaled time, multiply by 100 to get days)
        returns same shape broadcastable over channels
        """
        # Convert scaled time to actual days
        t_days = t * 100.0
        
        tc = torch.exp(self.log_tc).clamp(min=1.0)      # ensure >0
        alpha = 1.0 + 1e-3 + F.softplus(self.tilde_alpha)  # >1
        return torch.pow(1.0 + t_days / tc.unsqueeze(0), -alpha.unsqueeze(0))  # [..., state_dim]

    def forward_expert(self, x, t, z, w, s=None):
        """
        Compute this expert's output f_m([x, t, z]) using RFF weights w.

        x: [B, state_dim]
        t: [B, 1]
        z: [B, z_dim]
        w: [2*num_basis] or [B, 2*num_basis] after broadcasting
        s: [num_basis, rff_input_dim] - spectral frequencies (optional, will sample if not provided)
        returns: [B, state_dim]
        """
        B = x.shape[0]

        # If w has shape [2*num_basis], expand it over batch => [B, 2*num_basis]
        if w.dim() == 1:
            w = w.unsqueeze(0).expand(B, -1)  # shape [B, 2*num_basis]

        X_in = torch.cat([x, t, z], dim=1)  # [B, rff_input_dim]

        # Sample spectral frequencies if not provided
        if s is None:
            s = self.sample_spectral_frequencies(self.num_basis)  # [num_basis, rff_input_dim]

        # RFF with spectral mixture frequencies
        sim = torch.matmul(X_in, s.T)  # [B, num_basis]
        basis_sin = torch.sin(sim)
        basis_cos = torch.cos(sim)

        # Split w
        w_sin = w[:, :self.num_basis]   # [B, num_basis]
        w_cos = w[:, self.num_basis:]   # [B, num_basis]

        # Combine basis
        total_basis = -w_sin * basis_sin + w_cos * basis_cos  # [B, num_basis]

        # Map basis to [B, rff_input_dim], then select first state_dim
        f = torch.matmul(total_basis, s)  # => [B, rff_input_dim]
        f = f[:, :self.state_dim]         # => [B, state_dim]

        # Enforce positivity constraint with a smooth function (Softplus)
        # Softplus ≈ ReLU but with non-zero gradient for negative inputs,
        # which improves optimisation stability early in training.
        f = F.softplus(f)
        
        # Apply algebraic taper to ensure decline
        taper = self._taper(t)  # broadcast to [B, state_dim]
        f = f * taper
        
        return f

    def KL_divergence(self):
        """
        KL divergence q(w) || p(w) for the single global distribution.
        """
        b, L = self.compute_bL()
        nb_total = 2 * self.num_basis

        # Construct block diagonal for the [2n, 2n] covariance
        C = L @ L.transpose(0, 1)  # shape [n, n]
        zeros = torch.zeros_like(C)
        C_block = torch.cat([
            torch.cat([C, zeros], dim=-1),
            torch.cat([zeros, C], dim=-1)
        ], dim=0)  # => [2n, 2n]

        # Increase epsilon for better numerical stability
        eps = 1e-5
        eye = torch.eye(nb_total, device=b.device)
        C_block_stable = C_block + eps * eye

        # Use try-except to catch and handle numerical errors
        try:
            sign, logdetC = torch.linalg.slogdet(C_block_stable)
            if torch.isnan(logdetC) or torch.isinf(logdetC):
                # Fallback with even more regularization
                C_block_stable = C_block + 1e-3 * eye
                sign, logdetC = torch.linalg.slogdet(C_block_stable)
        except RuntimeError:
            # If decomposition fails, add more regularization
            C_block_stable = C_block + 1e-3 * eye
            sign, logdetC = torch.linalg.slogdet(C_block_stable)
            
        traceC = torch.diagonal(C_block_stable).sum()
        bsq = (b**2).sum()

        kl = 0.5 * (traceC + bsq - nb_total - logdetC)
        
        # Add a stability check
        if torch.isnan(kl) or torch.isinf(kl):
            # Return a small positive value instead
            print("Warning: KL divergence was NaN or Inf, using fallback value")
            kl = torch.tensor(1.0, device=b.device)
            
        return kl

    def KL_taper_params(self):
        """
        KL divergence for taper parameters.
        Prior: log_tc ~ N(log(365), 1.0)  # centered around 1 year
        Prior: tilde_alpha ~ N(0, 1.0)     # standard normal
        """
        # KL for log_tc: q(log_tc) || p(log_tc)
        # where q is point mass (MAP) and p is N(log(365), 1.0)
        log_tc_prior_mean = torch.log(torch.tensor(365.0, device=self.log_tc.device))
        log_tc_prior_var = 1.0
        kl_log_tc = 0.5 * torch.sum((self.log_tc - log_tc_prior_mean)**2 / log_tc_prior_var)
        
        # KL for tilde_alpha: q(tilde_alpha) || p(tilde_alpha)
        # where p is N(0, 1.0)
        kl_tilde_alpha = 0.5 * torch.sum(self.tilde_alpha**2)
        
        return kl_log_tc + kl_tilde_alpha

    def KL_noise_params(self):
        """
        KL divergence for noise covariance parameters.
        Prior: noise_unconstrained ~ N(0, 0.1)  # small noise prior
        """
        prior_var = 0.1
        kl_noise = 0.5 * torch.sum(self.noise_unconstrained**2 / prior_var)
        return kl_noise

    def KL_spectral_params(self):
        """
        KL divergence for spectral mixture parameters.
        Prior: mixture_means ~ N(0, 2.0)      # diverse frequencies
        Prior: log_mixture_variances ~ N(0, 1.0)
        Prior: log_mixture_weights ~ N(0, 1.0)
        """
        # KL for mixture means
        means_prior_var = 2.0
        kl_means = 0.5 * torch.sum(self.mixture_means**2 / means_prior_var)
        
        # KL for log mixture variances
        kl_log_var = 0.5 * torch.sum(self.log_mixture_variances**2)
        
        # KL for log mixture weights
        kl_log_weights = 0.5 * torch.sum(self.log_mixture_weights**2)
        
        return kl_means + kl_log_var + kl_log_weights


#######################################################
# 2) MixtureGP: Mixture-of-Experts with Gating Network
#######################################################
class MixtureGP(nn.Module):
    """
    Mixture of ExpertGP components with an input-dependent gating network,
    but each ExpertGP has a single global RFF weight distribution (not per-well).
    """
    def __init__(self, input_dim=3, z_dim=19, num_basis=16, num_experts=3, K=10):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.num_basis = num_basis
        self.num_experts = num_experts
        self.K = K

        # Create expert GP components (each has its own noise and global RFF)
        self.experts = nn.ModuleList([
            ExpertGP(state_dim=input_dim, z_dim=z_dim, num_basis=num_basis, K=K)
            for _ in range(num_experts)
        ])

        # Gating network: input = [x(3), t(1), z(19)] => 23
        gating_hidden = 50
        gating_in_dim = input_dim + 1 + z_dim
        self.gating_net = nn.Sequential(
            nn.Linear(gating_in_dim, gating_hidden),
            nn.ReLU(),
            nn.Linear(gating_hidden, num_experts)
        )

    def forward(self, t_scalar, x, z, return_mixture_weights=False):
        """
        ODE function: returns d/dt of x for time t_scalar.

        x: [B, input_dim]
        z: [B, z_dim]
        return_mixture_weights: if True, also return the mixture weights
        """
        B = x.shape[0]
        t_col = t_scalar.unsqueeze(0).expand(B).unsqueeze(1)  # [B, 1]

        # gating input => [B, input_dim+1+z_dim]
        gating_input = torch.cat([x, t_col, z], dim=1)
        logits = self.gating_net(gating_input)      # [B, num_experts]
        mix_weights = F.softmax(logits, dim=1)      # [B, num_experts]

        # Vectorised computation across the K Monte-Carlo weight samples to avoid
        # Python-side loops (≈ K× speed-up).
        expert_outputs = []
        for expert in self.experts:
            w_draws = expert.sample_weights()              # [K, 2*num_basis]
            
            # Sample spectral frequencies once per expert (shared across MC samples)
            s = expert.sample_spectral_frequencies(expert.num_basis)  # [num_basis, rff_input_dim]

            # Expand inputs along the K dimension
            B = x.shape[0]
            K = w_draws.shape[0]

            x_exp  = x.unsqueeze(0).expand(K, -1, -1)      # [K, B, D]
            t_exp  = t_col.unsqueeze(0).expand(K, -1, -1)  # [K, B, 1]
            z_exp  = z.unsqueeze(0).expand(K, -1, -1)      # [K, B, z]

            # Flatten the K and B dimensions so ExpertGP sees a standard batch
            x_flat = x_exp.reshape(K * B, self.input_dim)
            t_flat = t_exp.reshape(K * B, 1)
            z_flat = z_exp.reshape(K * B, self.z_dim)

            # Broadcast weight samples over batch dimension
            w_flat = w_draws.unsqueeze(1).expand(K, B, -1).reshape(K * B, -1)  # [K*B, 2*num_basis]

            f_flat = expert.forward_expert(x_flat, t_flat, z_flat, w_flat, s)      # [K*B, D]
            f_all  = f_flat.reshape(K, B, self.input_dim)                       # [K, B, D]

            f_m_mean = f_all.mean(dim=0)                                       # [B, D]
            expert_outputs.append(f_m_mean.unsqueeze(2))                        # [B, D, 1]

        # Blend experts by gating weights
        f_experts = torch.cat(expert_outputs, dim=2)    # [B, input_dim, num_experts]
        mix_wt_expanded = mix_weights.unsqueeze(1)      # [B, 1, num_experts]
        f_out = (f_experts * mix_wt_expanded).sum(dim=2)  # => [B, input_dim]

        if return_mixture_weights:
            return f_out, mix_weights
        return f_out

    def get_expert_covariances(self):
        """Returns the list of each expert's 3×3 noise covariance."""
        return [expert.get_noise_cov() for expert in self.experts]

    def get_merged_cov(self):
        """
        Return the average of the experts' covariances (simple placeholder).
        If you want gating-based combination, you'd do a weighted average
        of covariances by mixture weights, but this is a simpler approach.
        """
        Sigma_list = self.get_expert_covariances()
        return sum(Sigma_list) / len(Sigma_list)

    def neg_loglike(self, pred_x, target_x, use_merged_cov=True, expert_idx=0):
        """
        Negative log-likelihood given predictions, using either the average covariance
        or a single expert's covariance. 
        """
        B, T, D = pred_x.shape
        diff = (pred_x - target_x).reshape(-1, D)  # [B*T, 3]

        if use_merged_cov:
            Sigma = self.get_merged_cov()  # 3×3
        else:
            Sigma = self.experts[expert_idx].get_noise_cov()

        # Add larger epsilon to diagonal for improved numerical stability
        eps = 1e-4
        Sigma_stable = Sigma + eps * torch.eye(Sigma.shape[0], device=Sigma.device)
        
        # Use safer computation methods with error handling
        try:
            # Try computing with current stabilization
            Sigma_inv = torch.inverse(Sigma_stable)
            logdet_Sigma = torch.logdet(Sigma_stable)
            
            # Check for numerical issues
            if torch.isnan(logdet_Sigma) or torch.isinf(logdet_Sigma) or torch.isnan(Sigma_inv).any() or torch.isinf(Sigma_inv).any():
                # Try with more regularization
                Sigma_stable = Sigma + 1e-2 * torch.eye(Sigma.shape[0], device=Sigma.device)
                Sigma_inv = torch.inverse(Sigma_stable)
                logdet_Sigma = torch.logdet(Sigma_stable)
                
        except RuntimeError:
            # If computation fails, add more regularization
            print("Warning: Matrix inverse or logdet failed, using more regularization")
            Sigma_stable = Sigma + 1e-2 * torch.eye(Sigma.shape[0], device=Sigma.device)
            Sigma_inv = torch.inverse(Sigma_stable)
            logdet_Sigma = torch.logdet(Sigma_stable)
            
        # Compute quadratic term with clipping to prevent extreme values
        quad = torch.einsum("bi,ij,bj->b", diff, Sigma_inv, diff)
        quad = torch.clamp(quad, max=1e6)  # Prevent extreme values
        n = diff.shape[0]

        const = D * torch.log(torch.tensor(2.0 * 3.14159, device=diff.device))
        nll = 0.5 * (quad.sum() + n * (logdet_Sigma + const))
        
        # Safety check against NaN/Inf
        if torch.isnan(nll) or torch.isinf(nll):
            print("Warning: NLL computation resulted in NaN/Inf, using fallback value")
            nll = torch.tensor(1e3, device=diff.device)  # A large but finite value
            
        return nll

    def KL_divergence(self, kl_weights=None):
        """
        Sum the KL from each expert's parameters with configurable weights.
        
        Args:
            kl_weights: dict with keys 'rff', 'taper', 'noise', 'spectral', 'gating'
                       If None, uses default weights (all 1.0)
        """
        if kl_weights is None:
            kl_weights = {
                'rff': 1.0,
                'taper': 0.1,      # Start small to not disrupt training
                'noise': 0.1,      # Start small
                'spectral': 0.1,   # Start small
                'gating': 0.01     # Very small for gating network
            }
        
        kl_total = 0.0
        
        # Sum KL divergences from all experts
        for expert in self.experts:
            # RFF weights KL (existing)
            kl_total += kl_weights.get('rff', 1.0) * expert.KL_divergence()
            
            # Taper parameters KL
            kl_total += kl_weights.get('taper', 0.1) * expert.KL_taper_params()
            
            # Noise parameters KL
            kl_total += kl_weights.get('noise', 0.1) * expert.KL_noise_params()
            
            # Spectral mixture parameters KL
            kl_total += kl_weights.get('spectral', 0.1) * expert.KL_spectral_params()
        
        # Gating network KL (simple L2 regularization on weights)
        kl_gating = self.KL_gating_network()
        kl_total += kl_weights.get('gating', 0.01) * kl_gating
        
        return kl_total
    
    def KL_gating_network(self):
        """
        KL divergence for gating network parameters.
        Using simple L2 regularization (equivalent to Gaussian prior with mean 0).
        """
        kl = 0.0
        for name, param in self.gating_net.named_parameters():
            if 'weight' in name:  # Only regularize weights, not biases
                kl += 0.5 * torch.sum(param**2)
        return kl