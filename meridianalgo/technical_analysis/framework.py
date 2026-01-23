"""
Custom indicator development framework with automatic JIT compilation and validation.
"""

import ast
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import numba  # noqa: F401
    from numba import jit, njit, types  # noqa: F401
    from numba.typed import Dict as NumbaDict  # noqa: F401

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


logger = logging.getLogger(__name__)


@dataclass
class IndicatorMetadata:
    """Metadata for custom indicators."""

    name: str
    description: str
    category: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_columns: List[str] = field(default_factory=lambda: ["Close"])
    output_columns: List[str] = field(default_factory=list)
    min_periods: int = 1
    author: str = ""
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)


class IndicatorValidator:
    """Validator for custom indicators."""

    @staticmethod
    def validate_function(func: Callable) -> Tuple[bool, List[str]]:
        """Validate indicator function for JIT compilation compatibility."""
        errors = []

        try:
            # Get function source
            source = inspect.getsource(func)

            # Parse AST
            tree = ast.parse(source)

            # Check for unsupported constructs
            validator = ASTValidator()
            validator.visit(tree)
            errors.extend(validator.errors)

        except Exception as e:
            errors.append(f"Failed to parse function: {e}")

        return len(errors) == 0, errors

    @staticmethod
    def validate_parameters(parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate indicator parameters."""
        errors = []

        for name, value in parameters.items():
            if not isinstance(name, str):
                errors.append(f"Parameter name must be string: {name}")

            # Check for supported parameter types
            if not isinstance(value, (int, float, bool, str)):
                errors.append(f"Unsupported parameter type for {name}: {type(value)}")

        return len(errors) == 0, errors

    @staticmethod
    def validate_data_requirements(
        required_columns: List[str],
    ) -> Tuple[bool, List[str]]:
        """Validate data requirements."""
        errors = []

        valid_columns = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]

        for col in required_columns:
            if col not in valid_columns:
                errors.append(f"Invalid column requirement: {col}")

        return len(errors) == 0, errors


class ASTValidator(ast.NodeVisitor):
    """AST validator for Numba compatibility."""

    def __init__(self):
        self.errors = []
        self.unsupported_nodes = {
            ast.Import,
            ast.ImportFrom,
            ast.Global,
            ast.Nonlocal,
            ast.ClassDef,
            ast.AsyncFunctionDef,
            ast.AsyncWith,
            ast.AsyncFor,
        }

    def visit(self, node):
        if type(node) in self.unsupported_nodes:
            self.errors.append(f"Unsupported construct: {type(node).__name__}")

        # Check for unsupported function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                unsupported_funcs = {"print", "input", "open", "eval", "exec"}
                if node.func.id in unsupported_funcs:
                    self.errors.append(f"Unsupported function call: {node.func.id}")

        self.generic_visit(node)


class IndicatorCompiler:
    """Automatic JIT compiler for custom indicators."""

    def __init__(self):
        self.compiled_cache: Dict[str, Callable] = {}

    def compile_indicator(
        self, func: Callable, metadata: IndicatorMetadata
    ) -> Callable:
        """Compile indicator function with JIT."""
        if not NUMBA_AVAILABLE:
            logger.warning("Numba not available, using uncompiled function")
            return func

        # Check cache
        func_key = f"{metadata.name}_{hash(inspect.getsource(func))}"
        if func_key in self.compiled_cache:
            return self.compiled_cache[func_key]

        # Validate function
        is_valid, errors = IndicatorValidator.validate_function(func)
        if not is_valid:
            logger.warning(f"Function {metadata.name} not JIT compatible: {errors}")
            return func

        try:
            # Attempt JIT compilation
            compiled_func = njit(func)

            # Test compilation with dummy data
            test_data = self._generate_test_data(metadata)
            _ = compiled_func(**test_data)

            self.compiled_cache[func_key] = compiled_func
            logger.info(f"Successfully compiled indicator: {metadata.name}")
            return compiled_func

        except Exception as e:
            logger.warning(f"JIT compilation failed for {metadata.name}: {e}")
            return func

    def _generate_test_data(self, metadata: IndicatorMetadata) -> Dict[str, np.ndarray]:
        """Generate test data for compilation testing."""
        test_data = {}

        for col in metadata.required_columns:
            if col.lower() in ["open", "high", "low", "close"]:
                test_data[col.lower()] = np.random.random(100) * 100 + 50
            elif col.lower() == "volume":
                test_data[col.lower()] = np.random.randint(1000, 10000, 100).astype(
                    float
                )
            else:
                test_data[col.lower()] = np.random.random(100)

        # Add parameters
        for param, value in metadata.parameters.items():
            test_data[param] = value

        return test_data


class IndicatorBuilder:
    """Builder for creating custom indicators."""

    def __init__(self):
        self.compiler = IndicatorCompiler()
        self.validators = IndicatorValidator()

    def create_indicator(
        self, func: Callable, metadata: IndicatorMetadata, enable_jit: bool = True
    ) -> "CustomIndicatorFramework":
        """Create a custom indicator with validation and compilation."""

        # Validate parameters
        is_valid, errors = self.validators.validate_parameters(metadata.parameters)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {errors}")

        # Validate data requirements
        is_valid, errors = self.validators.validate_data_requirements(
            metadata.required_columns
        )
        if not is_valid:
            raise ValueError(f"Invalid data requirements: {errors}")

        # Compile if requested
        compiled_func = func
        if enable_jit:
            compiled_func = self.compiler.compile_indicator(func, metadata)

        return CustomIndicatorFramework(
            func=compiled_func, metadata=metadata, is_compiled=compiled_func != func
        )

    def create_from_source(
        self, source_code: str, metadata: IndicatorMetadata, enable_jit: bool = True
    ) -> "CustomIndicatorFramework":
        """Create indicator from source code string."""

        # Execute source code to get function
        namespace = {"np": np, "numpy": np}
        exec(source_code, namespace)

        # Find the function (assume it's the only function defined)
        func = None
        for name, obj in namespace.items():
            if (
                callable(obj)
                and not name.startswith("_")
                and name not in ["np", "numpy"]
            ):
                func = obj
                break

        if func is None:
            raise ValueError("No function found in source code")

        return self.create_indicator(func, metadata, enable_jit)


class CustomIndicatorFramework:
    """Enhanced custom indicator with framework features."""

    def __init__(
        self, func: Callable, metadata: IndicatorMetadata, is_compiled: bool = False
    ):
        self.func = func
        self.metadata = metadata
        self.is_compiled = is_compiled
        self.performance_stats = {
            "call_count": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "last_call_time": 0.0,
        }

    def calculate(self, data: pd.DataFrame, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate indicator with performance tracking."""
        import time

        start_time = time.time()

        # Validate input data
        self._validate_input(data)

        # Prepare inputs
        inputs = self._prepare_inputs(data, kwargs)

        # Calculate
        try:
            result = self.func(**inputs)

            # Format output
            formatted_result = self._format_output(result, data.index)

            # Update performance stats
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time)

            return formatted_result

        except Exception as e:
            logger.error(f"Error calculating {self.metadata.name}: {e}")
            raise

    def _validate_input(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        if len(data) < self.metadata.min_periods:
            raise ValueError(
                f"Insufficient data: need {self.metadata.min_periods}, got {len(data)}"
            )

        missing_columns = [
            col for col in self.metadata.required_columns if col not in data.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _prepare_inputs(
        self, data: pd.DataFrame, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare inputs for calculation."""
        inputs = {}

        # Add data columns
        for col in self.metadata.required_columns:
            inputs[col.lower()] = data[col].values

        # Add parameters
        for param, default_value in self.metadata.parameters.items():
            inputs[param] = kwargs.get(param, default_value)

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in inputs:
                inputs[key] = value

        return inputs

    def _format_output(
        self, result: Union[np.ndarray, float, Tuple], index: pd.Index
    ) -> Union[pd.Series, pd.DataFrame]:
        """Format calculation output."""
        if isinstance(result, tuple):
            # Multiple outputs
            if self.metadata.output_columns:
                columns = self.metadata.output_columns[: len(result)]
            else:
                columns = [f"{self.metadata.name}_{i}" for i in range(len(result))]

            data_dict = {}
            for i, (col, arr) in enumerate(zip(columns, result)):
                if isinstance(arr, np.ndarray):
                    data_dict[col] = arr
                else:
                    data_dict[col] = np.full(len(index), arr)

            return pd.DataFrame(data_dict, index=index)

        elif isinstance(result, np.ndarray):
            # Single array output
            if len(result) == 1:
                # Scalar result, broadcast to full length
                result = np.full(len(index), result[0])

            return pd.Series(result, index=index, name=self.metadata.name)

        else:
            # Scalar result
            result_array = np.full(len(index), result)
            return pd.Series(result_array, index=index, name=self.metadata.name)

    def _update_performance_stats(self, execution_time: float) -> None:
        """Update performance statistics."""
        self.performance_stats["call_count"] += 1
        self.performance_stats["total_time"] += execution_time
        self.performance_stats["avg_time"] = (
            self.performance_stats["total_time"] / self.performance_stats["call_count"]
        )
        self.performance_stats["last_call_time"] = execution_time

    def get_info(self) -> Dict[str, Any]:
        """Get indicator information."""
        return {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "category": self.metadata.category,
            "parameters": self.metadata.parameters,
            "required_columns": self.metadata.required_columns,
            "output_columns": self.metadata.output_columns,
            "min_periods": self.metadata.min_periods,
            "is_compiled": self.is_compiled,
            "performance_stats": self.performance_stats.copy(),
            "author": self.metadata.author,
            "version": self.metadata.version,
            "tags": self.metadata.tags,
        }

    def optimize_parameters(
        self,
        data: pd.DataFrame,
        parameter_ranges: Dict[str, Tuple[float, float]],
        optimization_metric: Callable = None,
        method: str = "grid_search",
    ) -> Dict[str, Any]:
        """Optimize indicator parameters."""
        if optimization_metric is None:
            # Default optimization metric (maximize absolute mean)
            def optimization_metric(x):
                return (
                    abs(x.mean()) if isinstance(x, pd.Series) else abs(x.mean().mean())
                )

        if method == "grid_search":
            return self._grid_search_optimization(
                data, parameter_ranges, optimization_metric
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def _grid_search_optimization(
        self,
        data: pd.DataFrame,
        parameter_ranges: Dict[str, Tuple[float, float]],
        optimization_metric: Callable,
    ) -> Dict[str, Any]:
        """Grid search parameter optimization."""
        best_params = self.metadata.parameters.copy()
        best_score = float("-inf")

        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(parameter_ranges)

        for params in param_combinations:
            try:
                # Calculate indicator with these parameters
                result = self.calculate(data, **params)

                # Calculate score
                score = optimization_metric(result)

                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                logger.warning(f"Parameter combination failed: {params}, error: {e}")
                continue

        return {
            "best_parameters": best_params,
            "best_score": best_score,
            "original_parameters": self.metadata.parameters,
        }

    def _generate_param_combinations(
        self, parameter_ranges: Dict[str, Tuple[float, float]], steps: int = 10
    ) -> List[Dict[str, Any]]:
        """Generate parameter combinations for grid search."""
        combinations = []

        # Simple grid search implementation
        param_names = list(parameter_ranges.keys())
        param_values = []

        for param_name in param_names:
            min_val, max_val = parameter_ranges[param_name]
            if isinstance(self.metadata.parameters.get(param_name, min_val), int):
                # Integer parameter
                values = list(range(int(min_val), int(max_val) + 1))
            else:
                # Float parameter
                values = np.linspace(min_val, max_val, steps).tolist()
            param_values.append(values)

        # Generate all combinations
        import itertools

        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)

        return combinations


class IndicatorRegistry:
    """Registry for managing custom indicators."""

    def __init__(self):
        self.indicators: Dict[str, CustomIndicatorFramework] = {}
        self.categories: Dict[str, List[str]] = {}
        self.builder = IndicatorBuilder()

    def register(self, indicator: CustomIndicatorFramework) -> None:
        """Register a custom indicator."""
        name = indicator.metadata.name

        if name in self.indicators:
            logger.warning(f"Overwriting existing indicator: {name}")

        self.indicators[name] = indicator

        # Update categories
        category = indicator.metadata.category
        if category not in self.categories:
            self.categories[category] = []

        if name not in self.categories[category]:
            self.categories[category].append(name)

        logger.info(f"Registered indicator: {name}")

    def unregister(self, name: str) -> None:
        """Unregister an indicator."""
        if name in self.indicators:
            indicator = self.indicators[name]
            category = indicator.metadata.category

            del self.indicators[name]

            if category in self.categories and name in self.categories[category]:
                self.categories[category].remove(name)

                # Remove empty categories
                if not self.categories[category]:
                    del self.categories[category]

            logger.info(f"Unregistered indicator: {name}")

    def get_indicator(self, name: str) -> Optional[CustomIndicatorFramework]:
        """Get an indicator by name."""
        return self.indicators.get(name)

    def list_indicators(self, category: str = None) -> List[str]:
        """List available indicators."""
        if category:
            return self.categories.get(category, [])
        else:
            return list(self.indicators.keys())

    def list_categories(self) -> List[str]:
        """List available categories."""
        return list(self.categories.keys())

    def search_indicators(self, query: str) -> List[str]:
        """Search indicators by name, description, or tags."""
        results = []
        query_lower = query.lower()

        for name, indicator in self.indicators.items():
            # Search in name
            if query_lower in name.lower():
                results.append(name)
                continue

            # Search in description
            if query_lower in indicator.metadata.description.lower():
                results.append(name)
                continue

            # Search in tags
            if any(query_lower in tag.lower() for tag in indicator.metadata.tags):
                results.append(name)
                continue

        return results

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of all indicators."""
        summary = {
            "total_indicators": len(self.indicators),
            "compiled_indicators": 0,
            "total_calls": 0,
            "total_time": 0.0,
            "slowest_indicators": [],
            "most_used_indicators": [],
        }

        indicator_stats = []

        for name, indicator in self.indicators.items():
            stats = indicator.performance_stats

            if indicator.is_compiled:
                summary["compiled_indicators"] += 1

            summary["total_calls"] += stats["call_count"]
            summary["total_time"] += stats["total_time"]

            indicator_stats.append(
                {
                    "name": name,
                    "avg_time": stats["avg_time"],
                    "call_count": stats["call_count"],
                    "total_time": stats["total_time"],
                }
            )

        # Sort by average time (slowest first)
        slowest = sorted(indicator_stats, key=lambda x: x["avg_time"], reverse=True)
        summary["slowest_indicators"] = slowest[:5]

        # Sort by call count (most used first)
        most_used = sorted(indicator_stats, key=lambda x: x["call_count"], reverse=True)
        summary["most_used_indicators"] = most_used[:5]

        return summary


# Global registry instance
indicator_registry = IndicatorRegistry()


# Decorator for easy indicator creation
def indicator(
    name: str,
    description: str = "",
    category: str = "custom",
    parameters: Dict[str, Any] = None,
    required_columns: List[str] = None,
    output_columns: List[str] = None,
    min_periods: int = 1,
    enable_jit: bool = True,
    auto_register: bool = True,
):
    """Decorator for creating custom indicators."""

    def decorator(func: Callable) -> CustomIndicatorFramework:
        metadata = IndicatorMetadata(
            name=name,
            description=description,
            category=category,
            parameters=parameters or {},
            required_columns=required_columns or ["Close"],
            output_columns=output_columns or [],
            min_periods=min_periods,
        )

        builder = IndicatorBuilder()
        custom_indicator = builder.create_indicator(func, metadata, enable_jit)

        if auto_register:
            indicator_registry.register(custom_indicator)

        return custom_indicator

    return decorator
