# src/data_processor.py

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    """
    Handles data loading, preprocessing, and preparation for training and evaluation.
    """
    def __init__(self, config):
        """
        Initialize the data processor with configuration.
        
        Args:
            config: Configuration dictionary containing data paths and parameters
        """
        self.config = config
        self.csv_file = config["paths"]["csv_file"]
        
        # Define columns for static features (z)
        self.z_columns = [
            'LateralLength_FT', 'FracStages', 'AverageStageSpacing_FT',
            'ProppantIntensity_LBSPerFT', 'FluidIntensity_BBLPerFT', 'Isopach_FT',
            'EffectivePorosity_PCT', 'TopOfZone_FT', 'GammaRay_API',
            'Resistivity_OHMSM', 'ClayVolume_PCT', 'WaterSaturation_PCT', 'PhiH_FT',
            'HCPV_PCT', 'TotalOrganicCarbon_WTPCT', 'BottomOfZone_FT',
            'TrueVerticalDepth_FT', 'Latitude', 'Longitude'
        ]
        
        # Time points in days
        self.days = [0, 30, 60, 90, 180, 360, 720, 1080]
        # Use normalized time (0-1) but with larger scale factor for better numerical stability
        # Scale by 0.01 instead of 1080.0 to keep values small but distinguishable
        self.t_vals = torch.tensor(self.days, dtype=torch.float32) * 0.01
        
        # Production columns by phase
        self.gas_cols = [
            'Prod0DaysGas_MCF_rs', 'Prod30DaysGas_MCF_rs', 'Prod60DaysGas_MCF_rs',
            'Prod90DaysGas_MCF_rs', 'Prod180DaysGas_MCF_rs', 'Prod360DaysGas_MCF_rs',
            'Prod720DaysGas_MCF_rs', 'Prod1080DaysGas_MCF_rs'
        ]
        self.oil_cols = [
            'Prod0DaysOil_BBL_rs', 'Prod30DaysOil_BBL_rs', 'Prod60DaysOil_BBL_rs',
            'Prod90DaysOil_BBL_rs', 'Prod180DaysOil_BBL_rs', 'Prod360DaysOil_BBL_rs',
            'Prod720DaysOil_BBL_rs', 'Prod1080DaysOil_BBL_rs'
        ]
        self.wat_cols = [
            'Prod0DaysWtr_BBL_rs', 'Prod30DaysWtr_BBL_rs', 'Prod60DaysWtr_BBL_rs',
            'Prod90DaysWtr_BBL_rs', 'Prod180DaysWtr_BBL_rs', 'Prod360DaysWtr_BBL_rs',
            'Prod720DaysWtr_BBL_rs', 'Prod1080DaysWtr_BBL_rs'
        ]
        
        # Scalers
        self.scaler_z = None

    def load_data(self):
        """
        Load data from CSV file.
        
        Returns:
            pandas.DataFrame: The loaded dataframe
        """
        return pd.read_csv(self.csv_file)
    
    def normalize_phase(self, values, phase_name, train_indices=None):
        """
        Normalize production data for a single phase by 360-day mean.
        
        Args:
            values: numpy array of production values [N, T]
            phase_name: Name of the phase ('gas', 'oil', 'wat')
            train_indices: Indices of training data for calculating mean
            
        Returns:
            numpy.ndarray: Normalized phase data
        """
        # Find the 360-day column index
        # Time points are: 0, 30, 60, 90, 180, 360, 720, 1080
        # So 360 days is at index 5
        day_360_idx = 5
        
        if train_indices is not None:
            # Use only training data to calculate mean
            values_360_train = values[train_indices, day_360_idx]  # 360-day values for training data
            non_zero_mask = values_360_train > 0
            if non_zero_mask.sum() > 0:
                mean_360 = values_360_train[non_zero_mask].mean()
            else:
                mean_360 = 1.0  # Fallback if all are zero
        else:
            # This shouldn't be used - always provide train_indices
            raise ValueError("train_indices must be provided to avoid data leakage")
        
        # Store the normalization scale (360-day mean from training data)
        setattr(self, f"{phase_name}_scale", mean_360)
        
        # Normalize by dividing by the 360-day mean
        return values / mean_360
    
    def prepare_data(self, test_size=None, random_state=None):
        """
        Prepare data for training and testing.
        
        Args:
            test_size: Fraction of data to use for testing (uses config if None)
            random_state: Random seed for reproducibility (uses config if None)
            
        Returns:
            tuple: (x_train, z_train, x_test, z_test, t_vals) tensors
        """
        # Use config values if not provided
        if test_size is None:
            test_size = self.config["training"]["validation_split"]
        if random_state is None:
            random_state = self.config["training"]["random_seed"]
        df = self.load_data()
        
        # Prepare static features (z) - but don't normalize yet
        z_data = df[self.z_columns].fillna(0).values.astype(np.float32)
        
        # Load raw production data (not normalized yet)
        gas_df = df[self.gas_cols].fillna(0)
        oil_df = df[self.oil_cols].fillna(0)
        wat_df = df[self.wat_cols].fillna(0)
        
        gas_raw = gas_df.values.astype(np.float32)  # shape [N, T]
        oil_raw = oil_df.values.astype(np.float32)
        wat_raw = wat_df.values.astype(np.float32)
        
        # Train/test split FIRST (before any normalization)
        N = gas_raw.shape[0]
        indices = np.arange(N)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        train_size = int((1 - test_size) * N)
        train_idx, test_idx = indices[:train_size], indices[train_size:]
        
        # Now normalize using ONLY training data statistics
        gas_data = self.normalize_phase(gas_raw, phase_name="gas", train_indices=train_idx)
        oil_data = self.normalize_phase(oil_raw, phase_name="oil", train_indices=train_idx)
        wat_data = self.normalize_phase(wat_raw, phase_name="wat", train_indices=train_idx)
        
        # Stack into 3D array
        x_array = np.stack([gas_data, oil_data, wat_data], axis=-1)  # [N, T, 3]
        x_tensor = torch.tensor(x_array, dtype=torch.float32)

        # Remember indices for mapping scale factors later
        self.train_idx = train_idx
        self.test_idx  = test_idx
        
        # Now fit StandardScaler ONLY on training data
        self.scaler_z = StandardScaler()
        z_train_norm = self.scaler_z.fit_transform(z_data[train_idx])  # Fit on train only
        z_test_norm = self.scaler_z.transform(z_data[test_idx])        # Transform test using train statistics
        
        # Convert to tensors
        z_train_tensor = torch.tensor(z_train_norm, dtype=torch.float32)
        z_test_tensor = torch.tensor(z_test_norm, dtype=torch.float32)
        
        # Store data as instance attributes for reuse
        self.x_train = x_tensor[train_idx]
        self.z_train = z_train_tensor
        self.x_test = x_tensor[test_idx]
        self.z_test = z_test_tensor

        # Corresponding per-well scale factors for de-normalisation
        scale_full = self.get_scale_tensor()
        # torch does not support indexing with numpy arrays directly
        self.scale_train = scale_full[torch.tensor(train_idx, dtype=torch.long)]
        self.scale_test  = scale_full[torch.tensor(test_idx, dtype=torch.long)]
        
        return self.x_train, self.z_train, self.x_test, self.z_test, self.t_vals
    
    def get_scaler(self):
        """Get the fitted z scaler."""
        return self.scaler_z
    
    def save_scaler(self, path):
        """Save the z scaler to a file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.scaler_z, f)
    
    def load_scaler(self, path):
        """Load a z scaler from a file."""
        import pickle
        with open(path, 'rb') as f:
            self.scaler_z = pickle.load(f)
        return self.scaler_z
        
    def get_data(self):
        """
        Returns the already prepared data without reprocessing.
        If data hasn't been prepared yet, calls prepare_data() to prepare it.
        
        Returns:
            tuple: (x_train, z_train, x_test, z_test, t_vals) tensors
        """
        # Check if data attributes exist
        if not hasattr(self, 'x_train') or not hasattr(self, 'z_train') or \
           not hasattr(self, 'x_test') or not hasattr(self, 'z_test'):
            # Data not prepared yet, call prepare_data
            return self.prepare_data()
        
        return self.x_train, self.z_train, self.x_test, self.z_test, self.t_vals

    # -------------------------------------------
    #  Scale utilities (for de-normalisation)
    # -------------------------------------------
    def get_scale_tensor(self):
        """Return scaling factors used in normalisation.

        Output: torch.Tensor of shape [N, 3] – columns are gas, oil, water.
        Note: Now returns the same scale for all wells in each phase.
        """
        if any(getattr(self, f"{p}_scale", None) is None for p in ["gas", "oil", "wat"]):
            raise RuntimeError("DataProcessor: scales not available – call prepare_data() first.")

        # Get the number of wells from x_train or x_test
        if hasattr(self, 'x_train'):
            n_train = self.x_train.shape[0]
            n_test = self.x_test.shape[0] if hasattr(self, 'x_test') else 0
            n_total = n_train + n_test
        else:
            raise RuntimeError("DataProcessor: data not prepared yet – call prepare_data() first.")
        
        # Create tensor with same scale for all wells
        gas = torch.full((n_total,), self.gas_scale, dtype=torch.float32)
        oil = torch.full((n_total,), self.oil_scale, dtype=torch.float32)
        wat = torch.full((n_total,), self.wat_scale, dtype=torch.float32)
        return torch.stack([gas, oil, wat], dim=1)  # [N, 3]