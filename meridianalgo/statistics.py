"""
Advanced Statistical Analysis Module for Financial Time Series

Comprehensive statistical analysis including risk metrics, correlation analysis,
cointegration testing, regime detection, hypothesis testing, and advanced
quantitative methods for financial applications.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Union, Tuple, Dict, List, Optional, Any
import warnings
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings('ignore')

# Check for optional dependencies
try:
    from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

class StatisticalArbitrage:
    """
    Advanced statistical arbitrage strategies and cointegration analysis.
    
    Features:
    - Cointegration testing (Engle-Granger, Johansen)
    - Pairs trading strategies
    - Mean reversion analysis
    - Statistical arbitrage portfolio construction
    - Hedge ratio calculation
    - Spread analysis
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with price data.
        
        Args:
            data: DataFrame with datetime index and price data
        """
        self.data = data
        self.returns = data.pct_change().dropna()
        self.pairs = None
        self.hedge_ratios = {}
    
    def calculate_rolling_correlation(self, window: int = 21) -> pd.DataFrame:
        """
        Calculate rolling correlation between all pairs of assets.
        
        Args:
            window: Rolling window size in periods (default: 21 for 1 month)
            
        Returns:
            DataFrame with rolling correlation coefficients
        """
        return self.returns.rolling(window=window).corr()
    
    def test_cointegration(
        self,
        x: Union[pd.Series, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        method: str = 'engle_granger',
        maxlag: int = 5
    ) -> Dict[str, float]:
        """
        Test for cointegration between two time series.
        
        Args:
            x: First time series
            y: Second time series
            method: Cointegration test method ('engle_granger', 'johansen')
            maxlag: Maximum lag for the test
            
        Returns:
            Dictionary with test results
        """
        if len(x) != len(y):
            raise ValueError("Input series must have the same length")
        
        if method == 'engle_granger' and STATSMODELS_AVAILABLE:
            score, pvalue, crit_values = coint(x, y, maxlag=maxlag)
            return {
                'test_statistic': score,
                'p_value': pvalue,
                'critical_values_1%': crit_values[0],
                'critical_values_5%': crit_values[1],
                'critical_values_10%': crit_values[2],
                'is_cointegrated': pvalue < 0.05
            }
        elif method == 'johansen' and STATSMODELS_AVAILABLE:
            # Johansen test for multiple series
            if isinstance(x, pd.Series):
                data_matrix = np.column_stack([x, y])
            else:
                data_matrix = np.column_stack([x, y])
            
            result = coint_johansen(data_matrix, det_order=0, k_ar_diff=maxlag)
            return {
                'trace_statistic': result.lr1,
                'max_eigen_statistic': result.lr2,
                'critical_values_trace': result.cvt,
                'critical_values_max_eigen': result.cvm,
                'eigenvalues': result.eig
            }
        else:
            # Fallback to simple correlation-based test
            correlation = np.corrcoef(x, y)[0, 1]
            return {
                'correlation': correlation,
                'is_cointegrated': abs(correlation) > 0.7
            }
    
    def find_cointegrated_pairs(
        self,
        significance_level: float = 0.05,
        method: str = 'engle_granger'
    ) -> List[Tuple[str, str, Dict[str, float]]]:
        """
        Find all cointegrated pairs in the dataset.
        
        Args:
            significance_level: Significance level for cointegration test
            method: Cointegration test method
            
        Returns:
            List of tuples with asset pairs and test results
        """
        cointegrated_pairs = []
        assets = self.data.columns
        
        for i, asset1 in enumerate(assets):
            for asset2 in assets[i+1:]:
                x = self.data[asset1].dropna()
                y = self.data[asset2].dropna()
                
                # Align series
                common_index = x.index.intersection(y.index)
                x_aligned = x.loc[common_index]
                y_aligned = y.loc[common_index]
                
                if len(x_aligned) > 50:  # Minimum data requirement
                    result = self.test_cointegration(x_aligned, y_aligned, method)
                    
                    if result.get('is_cointegrated', False):
                        cointegrated_pairs.append((asset1, asset2, result))
        
        self.pairs = cointegrated_pairs
        return cointegrated_pairs
    
    def calculate_hedge_ratio(
        self,
        y: pd.Series,
        x: pd.Series,
        method: str = 'ols'
    ) -> float:
        """
        Calculate hedge ratio for pairs trading.
        
        Args:
            y: Dependent variable (price of asset to hedge)
            x: Independent variable (price of hedging asset)
            method: Method for hedge ratio calculation ('ols', 'kalman')
            
        Returns:
            Hedge ratio
        """
        if method == 'ols':
            # Ordinary Least Squares
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            hedge_ratio = model.params[1]
        elif method == 'kalman':
            # Kalman filter (simplified)
            hedge_ratio = self._kalman_filter_hedge_ratio(y, x)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.hedge_ratios[(y.name, x.name)] = hedge_ratio
        return hedge_ratio
    
    def _kalman_filter_hedge_ratio(self, y: pd.Series, x: pd.Series) -> float:
        """Simplified Kalman filter for hedge ratio."""
        # Simplified implementation
        returns_y = y.pct_change().dropna()
        returns_x = x.pct_change().dropna()
        
        # Use rolling regression as approximation
        rolling_beta = returns_y.rolling(window=30).cov(returns_x) / returns_x.rolling(window=30).var()
        return rolling_beta.dropna().iloc[-1]
    
    def calculate_spread(
        self,
        y: pd.Series,
        x: pd.Series,
        hedge_ratio: Optional[float] = None
    ) -> pd.Series:
        """
        Calculate spread between two assets.
        
        Args:
            y: First asset price
            x: Second asset price
            hedge_ratio: Hedge ratio (calculated if None)
            
        Returns:
            Spread series
        """
        if hedge_ratio is None:
            hedge_ratio = self.calculate_hedge_ratio(y, x)
        
        spread = y - hedge_ratio * x
        return spread
    
    def test_mean_reversion(
        self,
        spread: pd.Series,
        window: int = 252
    ) -> Dict[str, float]:
        """
        Test if spread is mean-reverting.
        
        Args:
            spread: Spread series
            window: Window for calculations
            
        Returns:
            Dictionary with test results
        """
        # Augmented Dickey-Fuller test
        if STATSMODELS_AVAILABLE:
            adf_result = adfuller(spread.dropna())
            adf_statistic = adf_result[0]
            adf_pvalue = adf_result[1]
        else:
            # Simplified test using autocorrelation
            autocorr = spread.autocorr(lag=1)
            adf_statistic = (autocorr - 1) / np.sqrt(len(spread))
            adf_pvalue = 2 * (1 - stats.norm.cdf(abs(adf_statistic)))
        
        # Hurst exponent
        hurst = calculate_hurst_exponent(spread)
        
        # Half-life of mean reversion
        half_life = calculate_half_life(spread)
        
        return {
            'adf_statistic': adf_statistic,
            'adf_pvalue': adf_pvalue,
            'is_mean_reverting': adf_pvalue < 0.05,
            'hurst_exponent': hurst,
            'half_life': half_life
        }
    
    def generate_pairs_trading_signals(
        self,
        pair: Tuple[str, str],
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        lookback: int = 252
    ) -> pd.DataFrame:
        """
        Generate pairs trading signals for a given pair.
        
        Args:
            pair: Tuple of asset symbols
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            lookback: Lookback period for statistics
            
        Returns:
            DataFrame with trading signals
        """
        asset1, asset2 = pair
        prices1 = self.data[asset1].dropna()
        prices2 = self.data[asset2].dropna()
        
        # Align prices
        common_index = prices1.index.intersection(prices2.index)
        prices1 = prices1.loc[common_index]
        prices2 = prices2.loc[common_index]
        
        # Calculate hedge ratio and spread
        hedge_ratio = self.calculate_hedge_ratio(prices1, prices2)
        spread = self.calculate_spread(prices1, prices2, hedge_ratio)
        
        # Calculate z-score
        spread_mean = spread.rolling(window=lookback).mean()
        spread_std = spread.rolling(window=lookback).std()
        z_score = (spread - spread_mean) / spread_std
        
        # Generate signals
        signals = pd.DataFrame(index=spread.index)
        signals['spread'] = spread
        signals['z_score'] = z_score
        signals['hedge_ratio'] = hedge_ratio
        
        # Trading signals
        signals['long_signal'] = z_score < -entry_threshold
        signals['short_signal'] = z_score > entry_threshold
        signals['exit_signal'] = abs(z_score) < exit_threshold
        
        return signals


class AdvancedStatistics:
    """
    Advanced statistical analysis for financial time series.
    
    Features:
    - Regime detection
    - Volatility modeling
    - Correlation analysis
    - Hypothesis testing
    - Factor analysis
    - Clustering
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with financial data.
        
        Args:
            data: DataFrame with datetime index and price/return data
        """
        self.data = data
        self.returns = data.pct_change().dropna() if data.max().max() > 1 else data
    
    def detect_regimes(
        self,
        method: str = 'markov',
        n_regimes: int = 3,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Detect market regimes.
        
        Args:
            method: Detection method ('markov', 'clustering', 'threshold')
            n_regimes: Number of regimes
            window: Window for calculations
            
        Returns:
            DataFrame with regime labels
        """
        if method == 'markov':
            return self._markov_regime_detection(n_regimes, window)
        elif method == 'clustering':
            return self._clustering_regime_detection(n_regimes, window)
        elif method == 'threshold':
            return self._threshold_regime_detection(window)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _markov_regime_detection(self, n_regimes: int, window: int) -> pd.DataFrame:
        """Markov regime switching model (simplified)."""
        # Use volatility and returns as features
        features = pd.DataFrame()
        features['volatility'] = self.returns.rolling(window).std()
        features['returns'] = self.returns.rolling(window).mean()
        
        # Simple clustering as approximation to Markov model
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regime_labels = kmeans.fit_predict(features.dropna())
        
        regimes = pd.Series(index=features.dropna().index, data=regime_labels)
        return regimes.reindex(self.returns.index, method='ffill')
    
    def _clustering_regime_detection(self, n_regimes: int, window: int) -> pd.DataFrame:
        """Clustering-based regime detection."""
        features = pd.DataFrame()
        features['volatility'] = self.returns.rolling(window).std()
        features['skewness'] = self.returns.rolling(window).skew()
        features['kurtosis'] = self.returns.rolling(window).kurt()
        
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regime_labels = kmeans.fit_predict(features.dropna())
        
        regimes = pd.Series(index=features.dropna().index, data=regime_labels)
        return regimes.reindex(self.returns.index, method='ffill')
    
    def _threshold_regime_detection(self, window: int) -> pd.DataFrame:
        """Threshold-based regime detection."""
        volatility = self.returns.rolling(window).std()
        vol_threshold = volatility.quantile(0.7)
        
        regimes = pd.Series(0, index=self.returns.index)
        regimes[volatility > vol_threshold] = 1  # High volatility regime
        regimes[volatility > vol_threshold * 1.5] = 2  # Extreme volatility regime
        
        return regimes
    
    def model_volatility(
        self,
        method: str = 'garch',
        p: int = 1,
        q: int = 1,
        window: int = 252
    ) -> Dict[str, Any]:
        """
        Model volatility using GARCH or other methods.
        
        Args:
            method: Volatility model ('garch', 'ewma', 'historical')
            p: GARCH p parameter
            q: GARCH q parameter
            window: Window for historical volatility
            
        Returns:
            Dictionary with volatility model results
        """
        if method == 'garch' and ARCH_AVAILABLE:
            return self._garch_volatility(p, q)
        elif method == 'ewma':
            return self._ewma_volatility(window)
        elif method == 'historical':
            return self._historical_volatility(window)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _garch_volatility(self, p: int, q: int) -> Dict[str, Any]:
        """GARCH volatility modeling."""
        results = {}
        
        for asset in self.returns.columns:
            try:
                model = arch_model(self.returns[asset].dropna(), vol='Garch', p=p, q=q)
                fitted_model = model.fit(disp='off')
                
                results[asset] = {
                    'conditional_volatility': fitted_model.conditional_volatility,
                    'params': fitted_model.params,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic
                }
            except:
                # Fallback to EWMA
                ewma_vol = self.returns[asset].ewm(span=30).std()
                results[asset] = {
                    'conditional_volatility': ewma_vol,
                    'params': None,
                    'aic': np.nan,
                    'bic': np.nan
                }
        
        return results
    
    def _ewma_volatility(self, window: int) -> Dict[str, Any]:
        """EWMA volatility modeling."""
        results = {}
        
        for asset in self.returns.columns:
            ewma_vol = self.returns[asset].ewm(span=window).std()
            results[asset] = {
                'conditional_volatility': ewma_vol,
                'params': {'span': window},
                'aic': np.nan,
                'bic': np.nan
            }
        
        return results
    
    def _historical_volatility(self, window: int) -> Dict[str, Any]:
        """Historical volatility modeling."""
        results = {}
        
        for asset in self.returns.columns:
            hist_vol = self.returns[asset].rolling(window).std()
            results[asset] = {
                'conditional_volatility': hist_vol,
                'params': {'window': window},
                'aic': np.nan,
                'bic': np.nan
            }
        
        return results
    
    def calculate_correlation_matrix(
        self,
        method: str = 'pearson',
        min_periods: int = 252
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix with different methods.
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            min_periods: Minimum periods for calculation
            
        Returns:
            Correlation matrix
        """
        if method == 'pearson':
            return self.returns.corr(method='pearson', min_periods=min_periods)
        elif method == 'spearman':
            return self.returns.corr(method='spearman', min_periods=min_periods)
        elif method == 'kendall':
            return self.returns.corr(method='kendall', min_periods=min_periods)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_rolling_correlation(
        self,
        window: int = 63,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Calculate rolling correlations.
        
        Args:
            window: Rolling window size
            method: Correlation method
            
        Returns:
            Rolling correlation matrix
        """
        return self.returns.rolling(window).corr(method=method)
    
    def perform_factor_analysis(
        self,
        n_factors: int = 5,
        rotation: str = 'varimax'
    ) -> Dict[str, Any]:
        """
        Perform factor analysis on returns.
        
        Args:
            n_factors: Number of factors to extract
            rotation: Rotation method
            
        Returns:
            Dictionary with factor analysis results
        """
        from sklearn.decomposition import FactorAnalysis
        
        # Prepare data
        returns_clean = self.returns.dropna()
        
        # Fit factor analysis
        fa = FactorAnalysis(n_components=n_factors, rotation=rotation)
        factor_loadings = fa.fit_transform(returns_clean.T)
        
        # Create results
        factor_names = [f'Factor_{i+1}' for i in range(n_factors)]
        loadings_df = pd.DataFrame(
            fa.components_.T,
            index=returns_clean.columns,
            columns=factor_names
        )
        
        return {
            'loadings': loadings_df,
            'factor_scores': pd.DataFrame(factor_loadings, index=factor_names),
            'explained_variance': fa.noise_variance_,
            'n_factors': n_factors
        }
    
    def perform_pca_analysis(
        self,
        n_components: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform Principal Component Analysis.
        
        Args:
            n_components: Number of components (all if None)
            
        Returns:
            Dictionary with PCA results
        """
        # Prepare data
        returns_clean = self.returns.dropna()
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(returns_clean)
        
        # Create results
        component_names = [f'PC_{i+1}' for i in range(pca.n_components_)]
        
        return {
            'components': pd.DataFrame(
                pca.components_.T,
                index=returns_clean.columns,
                columns=component_names
            ),
            'explained_variance_ratio': pd.Series(
                pca.explained_variance_ratio_,
                index=component_names
            ),
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
            'scores': pd.DataFrame(
                principal_components,
                index=returns_clean.index,
                columns=component_names
            )
        }
    
    def hypothesis_test(
        self,
        test_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform various hypothesis tests.
        
        Args:
            test_type: Type of test ('normality', 'autocorrelation', 'stationarity')
            **kwargs: Test-specific parameters
            
        Returns:
            Dictionary with test results
        """
        if test_type == 'normality':
            return self._test_normality(**kwargs)
        elif test_type == 'autocorrelation':
            return self._test_autocorrelation(**kwargs)
        elif test_type == 'stationarity':
            return self._test_stationarity(**kwargs)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
    
    def _test_normality(self, asset: Optional[str] = None) -> Dict[str, Any]:
        """Test for normality of returns."""
        if asset is None:
            # Test all assets
            results = {}
            for col in self.returns.columns:
                results[col] = self._test_normality_single(col)
            return results
        else:
            return self._test_normality_single(asset)
    
    def _test_normality_single(self, asset: str) -> Dict[str, Any]:
        """Normality test for single asset."""
        returns_series = self.returns[asset].dropna()
        
        # Shapiro-Wilk test
        shapiro_stat, shapiro_p = stats.shapiro(returns_series)
        
        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(returns_series)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(returns_series, 'norm')
        
        return {
            'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_p},
            'jarque_bera': {'statistic': jb_stat, 'p_value': jb_p},
            'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_p},
            'is_normal': shapiro_p > 0.05 and jb_p > 0.05
        }
    
    def _test_autocorrelation(self, asset: str, lags: int = 20) -> Dict[str, Any]:
        """Test for autocorrelation."""
        returns_series = self.returns[asset].dropna()
        
        # Ljung-Box test
        lb_stat, lb_p = stats.acorr_ljungbox(returns_series, lags=[lags])
        
        # Calculate autocorrelation values
        autocorr_values = [returns_series.autocorr(lag=i) for i in range(1, lags + 1)]
        
        return {
            'ljung_box': {'statistic': lb_stat[0], 'p_value': lb_p[0]},
            'autocorrelations': autocorr_values,
            'has_autocorrelation': lb_p[0] < 0.05
        }
    
    def _test_stationarity(self, asset: str) -> Dict[str, Any]:
        """Test for stationarity."""
        returns_series = self.returns[asset].dropna()
        
        if STATSMODELS_AVAILABLE:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(returns_series)
            
            return {
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            }
        else:
            # Simplified test
            autocorr = returns_series.autocorr(lag=1)
            is_stationary = abs(autocorr) < 0.5
            
            return {
                'autocorrelation_lag1': autocorr,
                'is_stationary': is_stationary
            }


class RiskMetrics:
    """
    Comprehensive risk metrics calculation.
    
    Features:
    - Value at Risk (VaR)
    - Expected Shortfall (ES)
    - Drawdown analysis
    - Risk-adjusted returns
    - Stress testing
    """
    
    def __init__(self, returns: pd.DataFrame):
        """
        Initialize with returns data.
        
        Args:
            returns: DataFrame of returns
        """
        self.returns = returns
    
    def calculate_var(
        self,
        confidence_level: float = 0.95,
        method: str = 'historical',
        window: Optional[int] = None
    ) -> Union[float, pd.Series]:
        """
        Calculate Value at Risk.
        
        Args:
            confidence_level: Confidence level
            method: VaR method ('historical', 'gaussian', 'cornish_fisher')
            window: Rolling window size
            
        Returns:
            VaR value or series
        """
        alpha = 1 - confidence_level
        
        if window is None:
            # Static VaR
            if method == 'historical':
                return self.returns.quantile(alpha)
            elif method == 'gaussian':
                return self.returns.mean() + self.returns.std() * stats.norm.ppf(alpha)
            elif method == 'cornish_fisher':
                return self._cornish_fisher_var(alpha)
            else:
                raise ValueError(f"Unknown method: {method}")
        else:
            # Rolling VaR
            if method == 'historical':
                return self.returns.rolling(window).quantile(alpha)
            elif method == 'gaussian':
                rolling_mean = self.returns.rolling(window).mean()
                rolling_std = self.returns.rolling(window).std()
                return rolling_mean + rolling_std * stats.norm.ppf(alpha)
            else:
                raise ValueError(f"Rolling {method} VaR not implemented")
    
    def _cornish_fisher_var(self, alpha: float) -> float:
        """Cornish-Fisher expansion for VaR."""
        z = stats.norm.ppf(alpha)
        s = self.returns.skew()
        k = self.returns.kurtosis()
        
        cf_z = z + (s/6) * (z**2 - 1) + (k/24) * (z**3 - 3*z) - (s**2/36) * (2*z**3 - 5*z)
        
        return self.returns.mean() + self.returns.std() * cf_z
    
    def calculate_expected_shortfall(
        self,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> Union[float, pd.Series]:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            confidence_level: Confidence level
            method: ES method
            
        Returns:
            ES value or series
        """
        var = self.calculate_var(confidence_level, method)
        
        if isinstance(var, pd.Series):
            # Rolling ES
            es = pd.Series(index=self.returns.index, dtype=float)
            for i in range(len(self.returns)):
                if i >= len(var) - 1:
                    window_var = var.iloc[i]
                    window_returns = self.returns.iloc[max(0, i-251):i+1]
                    es.iloc[i] = window_returns[window_returns <= window_var].mean()
            return es.dropna()
        else:
            # Static ES
            return self.returns[self.returns <= var].mean()
    
    def calculate_drawdowns(self) -> Dict[str, pd.Series]:
        """
        Calculate drawdown metrics.
        
        Returns:
            Dictionary with drawdown series
        """
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        return {
            'drawdown': drawdown,
            'running_max': running_max,
            'cumulative_returns': cumulative
        }
    
    def calculate_risk_adjusted_returns(
        self,
        risk_free_rate: float = 0.0
    ) -> Dict[str, pd.Series]:
        """
        Calculate risk-adjusted return metrics.
        
        Args:
            risk_free_rate: Risk-free rate
            
        Returns:
            Dictionary with risk-adjusted metrics
        """
        # Annualized metrics
        annualized_return = self.returns.mean() * 252
        annualized_volatility = self.returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        
        # Sortino ratio
        downside_returns = self.returns[self.returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - risk_free_rate) / downside_volatility
        
        # Calmar ratio
        drawdown = self.calculate_drawdowns()['drawdown']
        max_drawdown = drawdown.min()
        calmar_ratio = annualized_return / abs(max_drawdown)
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility
        }


# Utility functions
def calculate_correlation_matrix(
    returns: pd.DataFrame,
    method: str = 'pearson',
    min_periods: int = 252
) -> pd.DataFrame:
    """
    Calculate correlation matrix.
    
    Args:
        returns: DataFrame of returns
        method: Correlation method
        min_periods: Minimum periods
        
    Returns:
        Correlation matrix
    """
    return returns.corr(method=method, min_periods=min_periods)


def calculate_rolling_correlation(
    returns: pd.DataFrame,
    window: int = 63,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Calculate rolling correlations.
    
    Args:
        returns: DataFrame of returns
        window: Rolling window
        method: Correlation method
        
    Returns:
        Rolling correlation matrix
    """
    return returns.rolling(window).corr(method=method)


def calculate_hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst exponent for time series.
    
    Args:
        series: Time series
        max_lag: Maximum lag
        
    Returns:
        Hurst exponent
    """
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(series.values[i:], series.values[:-i]))) for i in lags]
    
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = poly[0] * 2.0
    
    return hurst


def calculate_half_life(series: pd.Series) -> float:
    """
    Calculate half-life of mean reversion.
    
    Args:
        series: Time series
        
    Returns:
        Half-life in periods
    """
    # Calculate lagged series
    lagged = series.shift(1).dropna()
    current = series.loc[lagged.index]
    
    # Regression of current on lagged
    slope = np.polyfit(lagged, current, 1)[0]
    
    # Calculate half-life
    half_life = -np.log(2) / np.log(slope) if slope > 0 and slope < 1 else np.inf
    
    return half_life


def perform_statistical_tests(
    returns: pd.DataFrame,
    tests: List[str] = ['normality', 'autocorrelation', 'stationarity']
) -> Dict[str, Any]:
    """
    Perform comprehensive statistical tests.
    
    Args:
        returns: DataFrame of returns
        tests: List of tests to perform
        
    Returns:
        Dictionary with test results
    """
    stats_analyzer = AdvancedStatistics(returns)
    results = {}
    
    for test in tests:
        results[test] = stats_analyzer.hypothesis_test(test)
    
    return results


# Export main classes and functions
__all__ = [
    'StatisticalArbitrage',
    'AdvancedStatistics',
    'RiskMetrics',
    'calculate_correlation_matrix',
    'calculate_rolling_correlation',
    'calculate_hurst_exponent',
    'calculate_half_life',
    'perform_statistical_tests',
    'calculate_value_at_risk',
    'calculate_expected_shortfall',
    'calculate_rolling_volatility',
    'calculate_metrics'
]