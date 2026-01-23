"""
Interactive visualization system with Plotly integration for technical analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import plotly.express as px  # noqa: F401
    import plotly.graph_objects as go
    import plotly.offline as pyo  # noqa: F401
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Visualization features will be limited.")

try:
    import dash
    import dash_bootstrap_components as dbc
    from dash import Input, Output, State, callback, dcc, html  # noqa: F401

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False


class TechnicalChart:
    """Interactive technical analysis chart with Plotly."""

    def __init__(
        self,
        title: str = "Technical Analysis Chart",
        width: int = 1200,
        height: int = 800,
    ):
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for visualization features")

        self.title = title
        self.width = width
        self.height = height
        self.fig = None
        self.data = None
        self.indicators = {}
        self.patterns = []
        self.annotations = []

    def set_data(self, data: pd.DataFrame) -> "TechnicalChart":
        """Set OHLCV data for the chart."""
        required_columns = ["Open", "High", "Low", "Close"]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.data = data.copy()
        return self

    def add_candlestick(
        self,
        name: str = "Price",
        increasing_color: str = "#00ff00",
        decreasing_color: str = "#ff0000",
    ) -> "TechnicalChart":
        """Add candlestick chart."""
        if self.data is None:
            raise ValueError("Data must be set before adding candlestick chart")

        if self.fig is None:
            self.fig = go.Figure()

        candlestick = go.Candlestick(
            x=self.data.index,
            open=self.data["Open"],
            high=self.data["High"],
            low=self.data["Low"],
            close=self.data["Close"],
            name=name,
            increasing_line_color=increasing_color,
            decreasing_line_color=decreasing_color,
            increasing_fillcolor=increasing_color,
            decreasing_fillcolor=decreasing_color,
        )

        self.fig.add_trace(candlestick)
        return self

    def add_ohlc(
        self,
        name: str = "Price",
        increasing_color: str = "#00ff00",
        decreasing_color: str = "#ff0000",
    ) -> "TechnicalChart":
        """Add OHLC bar chart."""
        if self.data is None:
            raise ValueError("Data must be set before adding OHLC chart")

        if self.fig is None:
            self.fig = go.Figure()

        ohlc = go.Ohlc(
            x=self.data.index,
            open=self.data["Open"],
            high=self.data["High"],
            low=self.data["Low"],
            close=self.data["Close"],
            name=name,
            increasing_line_color=increasing_color,
            decreasing_line_color=decreasing_color,
        )

        self.fig.add_trace(ohlc)
        return self

    def add_line(
        self,
        data: Union[pd.Series, np.ndarray],
        name: str,
        color: str = None,
        line_width: int = 2,
        line_dash: str = None,
        secondary_y: bool = False,
    ) -> "TechnicalChart":
        """Add line chart."""
        if self.fig is None:
            self.fig = go.Figure()

        x_data = self.data.index if self.data is not None else range(len(data))

        line = go.Scatter(
            x=x_data,
            y=data,
            mode="lines",
            name=name,
            line=dict(color=color, width=line_width, dash=line_dash),
            yaxis="y2" if secondary_y else "y",
        )

        self.fig.add_trace(line)
        return self

    def add_indicator(
        self,
        indicator_data: Union[pd.Series, pd.DataFrame],
        name: str,
        colors: List[str] = None,
        secondary_y: bool = False,
    ) -> "TechnicalChart":
        """Add technical indicator to chart."""
        if isinstance(indicator_data, pd.Series):
            self.add_line(
                indicator_data,
                name,
                color=colors[0] if colors else None,
                secondary_y=secondary_y,
            )
        elif isinstance(indicator_data, pd.DataFrame):
            default_colors = ["blue", "red", "green", "orange", "purple"]
            for i, col in enumerate(indicator_data.columns):
                color = (
                    colors[i]
                    if colors and i < len(colors)
                    else default_colors[i % len(default_colors)]
                )
                self.add_line(
                    indicator_data[col],
                    f"{name}_{col}",
                    color=color,
                    secondary_y=secondary_y,
                )

        self.indicators[name] = indicator_data
        return self

    def add_volume(
        self,
        volume_data: pd.Series = None,
        name: str = "Volume",
        color: str = "rgba(0,100,80,0.5)",
    ) -> "TechnicalChart":
        """Add volume bars."""
        if (
            volume_data is None
            and self.data is not None
            and "Volume" in self.data.columns
        ):
            volume_data = self.data["Volume"]

        if volume_data is None:
            logger.warning("No volume data available")
            return self

        if self.fig is None:
            self.fig = go.Figure()

        x_data = self.data.index if self.data is not None else range(len(volume_data))

        volume_bars = go.Bar(
            x=x_data, y=volume_data, name=name, marker_color=color, yaxis="y2"
        )

        self.fig.add_trace(volume_bars)
        return self

    def add_bollinger_bands(
        self,
        bb_data: pd.DataFrame,
        name: str = "Bollinger Bands",
        fill_color: str = "rgba(0,100,80,0.2)",
    ) -> "TechnicalChart":
        """Add Bollinger Bands."""
        if "BB_Upper" not in bb_data.columns or "BB_Lower" not in bb_data.columns:
            raise ValueError(
                "Bollinger Bands data must contain 'BB_Upper' and 'BB_Lower' columns"
            )

        x_data = self.data.index if self.data is not None else bb_data.index

        # Upper band
        self.fig.add_trace(
            go.Scatter(
                x=x_data,
                y=bb_data["BB_Upper"],
                mode="lines",
                name=f"{name} Upper",
                line=dict(color="rgba(0,100,80,0)", width=0),
                showlegend=False,
            )
        )

        # Lower band with fill
        self.fig.add_trace(
            go.Scatter(
                x=x_data,
                y=bb_data["BB_Lower"],
                mode="lines",
                name=name,
                line=dict(color="rgba(0,100,80,0)", width=0),
                fill="tonexty",
                fillcolor=fill_color,
            )
        )

        # Middle band if available
        if "BB_Middle" in bb_data.columns:
            self.add_line(
                bb_data["BB_Middle"],
                f"{name} Middle",
                color="rgba(0,100,80,0.8)",
                line_dash="dash",
            )

        return self

    def add_pattern_annotation(
        self, pattern_result, annotation_color: str = "red"
    ) -> "TechnicalChart":
        """Add pattern annotation to chart."""
        if self.data is None:
            return self

        start_date = self.data.index[pattern_result.start_index]
        end_date = self.data.index[pattern_result.end_index]

        # Get price range for annotation positioning
        price_range = self.data.loc[start_date:end_date]
        y_position = price_range["High"].max() * 1.02

        annotation = dict(
            x=start_date,
            y=y_position,
            text=f"{pattern_result.pattern_name}<br>Conf: {pattern_result.confidence:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=annotation_color,
            arrowwidth=2,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=annotation_color,
            borderwidth=1,
            font=dict(size=10),
        )

        self.annotations.append(annotation)
        return self

    def add_support_resistance(
        self, levels: List[float], colors: List[str] = None, line_dash: str = "dash"
    ) -> "TechnicalChart":
        """Add support/resistance levels."""
        if self.data is None:
            return self

        default_colors = ["red", "green", "blue", "orange"]

        for i, level in enumerate(levels):
            color = (
                colors[i]
                if colors and i < len(colors)
                else default_colors[i % len(default_colors)]
            )

            self.fig.add_hline(
                y=level,
                line_dash=line_dash,
                line_color=color,
                annotation_text=f"Level {level:.2f}",
                annotation_position="bottom right",
            )

        return self

    def add_trend_line(
        self,
        start_point: Tuple[datetime, float],
        end_point: Tuple[datetime, float],
        name: str = "Trend Line",
        color: str = "blue",
        line_width: int = 2,
    ) -> "TechnicalChart":
        """Add trend line."""
        self.fig.add_shape(
            type="line",
            x0=start_point[0],
            y0=start_point[1],
            x1=end_point[0],
            y1=end_point[1],
            line=dict(color=color, width=line_width),
            name=name,
        )

        return self

    def create_subplot_chart(
        self, subplot_config: List[Dict[str, Any]]
    ) -> "TechnicalChart":
        """Create chart with multiple subplots."""
        rows = len(subplot_config)

        # Calculate row heights
        row_heights = [config.get("height", 1.0) for config in subplot_config]

        # Create subplot titles
        subplot_titles = [
            config.get("title", f"Subplot {i + 1}")
            for i, config in enumerate(subplot_config)
        ]

        self.fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=row_heights,
            subplot_titles=subplot_titles,
        )

        return self

    def add_to_subplot(self, trace, row: int, col: int = 1) -> "TechnicalChart":
        """Add trace to specific subplot."""
        if self.fig is None:
            raise ValueError("Must create subplot chart first")

        self.fig.add_trace(trace, row=row, col=col)
        return self

    def update_layout(self, **kwargs) -> "TechnicalChart":
        """Update chart layout."""
        if self.fig is None:
            self.fig = go.Figure()

        default_layout = dict(
            title=self.title,
            width=self.width,
            height=self.height,
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            hovermode="x unified",
        )

        # Merge with custom layout options
        layout_options = {**default_layout, **kwargs}

        self.fig.update_layout(**layout_options)

        # Add annotations
        if self.annotations:
            self.fig.update_layout(annotations=self.annotations)

        return self

    def show(self, **kwargs) -> None:
        """Display the chart."""
        if self.fig is None:
            raise ValueError("No chart data to display")

        self.fig.show(**kwargs)

    def to_html(self, filename: str = None, **kwargs) -> str:
        """Export chart to HTML."""
        if self.fig is None:
            raise ValueError("No chart data to export")

        if filename:
            self.fig.write_html(filename, **kwargs)
            return filename
        else:
            return self.fig.to_html(**kwargs)

    def to_image(self, filename: str, format: str = "png", **kwargs) -> str:
        """Export chart to image."""
        if self.fig is None:
            raise ValueError("No chart data to export")

        self.fig.write_image(filename, format=format, **kwargs)
        return filename


class ChartTemplate:
    """Pre-defined chart templates for common technical analysis scenarios."""

    @staticmethod
    def create_basic_candlestick(
        data: pd.DataFrame, title: str = "Candlestick Chart"
    ) -> TechnicalChart:
        """Create basic candlestick chart."""
        chart = TechnicalChart(title=title)
        chart.set_data(data)
        chart.add_candlestick()

        if "Volume" in data.columns:
            chart.create_subplot_chart(
                [{"title": "Price", "height": 0.7}, {"title": "Volume", "height": 0.3}]
            )

            # Add candlestick to first subplot
            candlestick = go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Price",
            )
            chart.add_to_subplot(candlestick, row=1)

            # Add volume to second subplot
            volume_bars = go.Bar(
                x=data.index,
                y=data["Volume"],
                name="Volume",
                marker_color="rgba(0,100,80,0.5)",
            )
            chart.add_to_subplot(volume_bars, row=2)

        chart.update_layout()
        return chart

    @staticmethod
    def create_moving_average_chart(
        data: pd.DataFrame,
        ma_periods: List[int] = [20, 50, 200],
        title: str = "Moving Averages Chart",
    ) -> TechnicalChart:
        """Create chart with moving averages."""
        chart = TechnicalChart(title=title)
        chart.set_data(data)
        chart.add_candlestick()

        # Add moving averages
        colors = ["blue", "red", "green", "orange", "purple"]
        for i, period in enumerate(ma_periods):
            ma = data["Close"].rolling(window=period).mean()
            color = colors[i % len(colors)]
            chart.add_line(ma, f"MA{period}", color=color)

        chart.update_layout()
        return chart

    @staticmethod
    def create_bollinger_bands_chart(
        data: pd.DataFrame,
        bb_period: int = 20,
        bb_std: float = 2.0,
        title: str = "Bollinger Bands Chart",
    ) -> TechnicalChart:
        """Create chart with Bollinger Bands."""
        chart = TechnicalChart(title=title)
        chart.set_data(data)
        chart.add_candlestick()

        # Calculate Bollinger Bands
        sma = data["Close"].rolling(window=bb_period).mean()
        std = data["Close"].rolling(window=bb_period).std()

        bb_data = pd.DataFrame(
            {
                "BB_Upper": sma + (std * bb_std),
                "BB_Middle": sma,
                "BB_Lower": sma - (std * bb_std),
            },
            index=data.index,
        )

        chart.add_bollinger_bands(bb_data)
        chart.update_layout()
        return chart

    @staticmethod
    def create_rsi_chart(
        data: pd.DataFrame, rsi_period: int = 14, title: str = "RSI Chart"
    ) -> TechnicalChart:
        """Create chart with RSI indicator."""
        chart = TechnicalChart(title=title)
        chart.set_data(data)

        # Calculate RSI
        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Create subplots
        chart.create_subplot_chart(
            [{"title": "Price", "height": 0.7}, {"title": "RSI", "height": 0.3}]
        )

        # Add candlestick to first subplot
        candlestick = go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
        )
        chart.add_to_subplot(candlestick, row=1)

        # Add RSI to second subplot
        rsi_line = go.Scatter(
            x=data.index, y=rsi, mode="lines", name="RSI", line=dict(color="purple")
        )
        chart.add_to_subplot(rsi_line, row=2)

        # Add RSI reference lines
        chart.fig.add_hline(y=70, line_dash="dash", line_color="red", row=2)
        chart.fig.add_hline(y=30, line_dash="dash", line_color="green", row=2)
        chart.fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2)

        chart.update_layout()
        return chart


class InteractiveDashboard:
    """Interactive dashboard using Dash for real-time technical analysis."""

    def __init__(self, title: str = "Technical Analysis Dashboard"):
        if not DASH_AVAILABLE:
            raise ImportError("Dash is required for interactive dashboard features")

        self.title = title
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.data_sources = {}
        self.indicators = {}
        self.charts = {}

        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(self.title, className="text-center mb-4"),
                            ],
                            width=12,
                        )
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Chart Controls"),
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Symbol"),
                                                                dcc.Dropdown(
                                                                    id="symbol-dropdown",
                                                                    options=[],
                                                                    value=None,
                                                                    placeholder="Select symbol",
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Chart Type"),
                                                                dcc.Dropdown(
                                                                    id="chart-type-dropdown",
                                                                    options=[
                                                                        {
                                                                            "label": "Candlestick",
                                                                            "value": "candlestick",
                                                                        },
                                                                        {
                                                                            "label": "OHLC",
                                                                            "value": "ohlc",
                                                                        },
                                                                        {
                                                                            "label": "Line",
                                                                            "value": "line",
                                                                        },
                                                                    ],
                                                                    value="candlestick",
                                                                ),
                                                            ],
                                                            width=6,
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Indicators"),
                                                                dcc.Checklist(
                                                                    id="indicators-checklist",
                                                                    options=[
                                                                        {
                                                                            "label": "Moving Average (20)",
                                                                            "value": "ma20",
                                                                        },
                                                                        {
                                                                            "label": "Moving Average (50)",
                                                                            "value": "ma50",
                                                                        },
                                                                        {
                                                                            "label": "Bollinger Bands",
                                                                            "value": "bb",
                                                                        },
                                                                        {
                                                                            "label": "RSI",
                                                                            "value": "rsi",
                                                                        },
                                                                        {
                                                                            "label": "MACD",
                                                                            "value": "macd",
                                                                        },
                                                                    ],
                                                                    value=[],
                                                                    inline=True,
                                                                ),
                                                            ],
                                                            width=12,
                                                        )
                                                    ],
                                                    className="mb-3",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Button(
                                                                    "Update Chart",
                                                                    id="update-button",
                                                                    color="primary",
                                                                    className="w-100",
                                                                )
                                                            ],
                                                            width=12,
                                                        )
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ]
                                )
                            ],
                            width=3,
                        ),
                        dbc.Col(
                            [dcc.Graph(id="main-chart", style={"height": "600px"})],
                            width=9,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(
                                    id="indicator-chart", style={"height": "300px"}
                                )
                            ],
                            width=12,
                        )
                    ]
                ),
            ],
            fluid=True,
        )

    def _setup_callbacks(self):
        """Setup dashboard callbacks."""

        @self.app.callback(
            [Output("main-chart", "figure"), Output("indicator-chart", "figure")],
            [Input("update-button", "n_clicks")],
            [
                State("symbol-dropdown", "value"),
                State("chart-type-dropdown", "value"),
                State("indicators-checklist", "value"),
            ],
        )
        def update_charts(n_clicks, symbol, chart_type, indicators):
            if not symbol or symbol not in self.data_sources:
                return {}, {}

            data = self.data_sources[symbol]

            # Create main chart
            main_fig = self._create_main_chart(data, chart_type, indicators)

            # Create indicator chart
            indicator_fig = self._create_indicator_chart(data, indicators)

            return main_fig, indicator_fig

    def _create_main_chart(
        self, data: pd.DataFrame, chart_type: str, indicators: List[str]
    ):
        """Create main price chart."""
        fig = go.Figure()

        # Add price data
        if chart_type == "candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"],
                    name="Price",
                )
            )
        elif chart_type == "ohlc":
            fig.add_trace(
                go.Ohlc(
                    x=data.index,
                    open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"],
                    name="Price",
                )
            )
        else:  # line
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data["Close"], mode="lines", name="Close Price"
                )
            )

        # Add indicators
        if "ma20" in indicators:
            ma20 = data["Close"].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=ma20,
                    mode="lines",
                    name="MA20",
                    line=dict(color="blue"),
                )
            )

        if "ma50" in indicators:
            ma50 = data["Close"].rolling(window=50).mean()
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=ma50,
                    mode="lines",
                    name="MA50",
                    line=dict(color="red"),
                )
            )

        if "bb" in indicators:
            sma = data["Close"].rolling(window=20).mean()
            std = data["Close"].rolling(window=20).std()

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=sma + (std * 2),
                    mode="lines",
                    name="BB Upper",
                    line=dict(color="rgba(0,100,80,0)", width=0),
                    showlegend=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=sma - (std * 2),
                    mode="lines",
                    name="Bollinger Bands",
                    line=dict(color="rgba(0,100,80,0)", width=0),
                    fill="tonexty",
                    fillcolor="rgba(0,100,80,0.2)",
                )
            )

        fig.update_layout(
            title="Price Chart",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            height=600,
        )

        return fig

    def _create_indicator_chart(self, data: pd.DataFrame, indicators: List[str]):
        """Create indicator chart."""
        fig = go.Figure()

        if "rsi" in indicators:
            # Calculate RSI
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=rsi,
                    mode="lines",
                    name="RSI",
                    line=dict(color="purple"),
                )
            )

            fig.add_hline(y=70, line_dash="dash", line_color="red")
            fig.add_hline(y=30, line_dash="dash", line_color="green")
            fig.add_hline(y=50, line_dash="dot", line_color="gray")

        if "macd" in indicators:
            # Calculate MACD
            ema12 = data["Close"].ewm(span=12).mean()
            ema26 = data["Close"].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=macd,
                    mode="lines",
                    name="MACD",
                    line=dict(color="blue"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=signal,
                    mode="lines",
                    name="Signal",
                    line=dict(color="red"),
                )
            )

            fig.add_trace(
                go.Bar(x=data.index, y=histogram, name="Histogram", marker_color="gray")
            )

        fig.update_layout(
            title="Technical Indicators", template="plotly_white", height=300
        )

        return fig

    def add_data_source(self, symbol: str, data: pd.DataFrame):
        """Add data source for dashboard."""
        self.data_sources[symbol] = data

        # Update symbol dropdown options
        [{"label": symbol, "value": symbol} for symbol in self.data_sources.keys()]

        # This would need to be handled differently in a real implementation
        # as we can't directly update component properties from here
        logger.info(f"Added data source for {symbol}")

    def run(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = True):
        """Run the dashboard."""
        self.app.run_server(host=host, port=port, debug=debug)


class ChartAnnotationTool:
    """Tool for adding annotations and drawings to charts."""

    def __init__(self, chart: TechnicalChart):
        self.chart = chart
        self.annotations = []
        self.drawings = []

    def add_text_annotation(
        self, x, y, text: str, color: str = "black", size: int = 12
    ):
        """Add text annotation."""
        annotation = dict(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font=dict(color=color, size=size),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color,
            borderwidth=1,
        )

        self.annotations.append(annotation)
        self.chart.annotations.extend([annotation])
        return self

    def add_arrow_annotation(
        self, x, y, text: str, arrow_color: str = "red", text_color: str = "black"
    ):
        """Add arrow annotation."""
        annotation = dict(
            x=x,
            y=y,
            text=text,
            showarrow=True,
            arrowhead=2,
            arrowcolor=arrow_color,
            arrowwidth=2,
            font=dict(color=text_color),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=arrow_color,
            borderwidth=1,
        )

        self.annotations.append(annotation)
        self.chart.annotations.extend([annotation])
        return self

    def add_horizontal_line(
        self, y: float, color: str = "blue", line_dash: str = "dash", text: str = None
    ):
        """Add horizontal line."""
        self.chart.fig.add_hline(
            y=y,
            line_dash=line_dash,
            line_color=color,
            annotation_text=text or f"Level {y:.2f}",
            annotation_position="bottom right",
        )
        return self

    def add_vertical_line(
        self, x, color: str = "blue", line_dash: str = "dash", text: str = None
    ):
        """Add vertical line."""
        self.chart.fig.add_vline(
            x=x,
            line_dash=line_dash,
            line_color=color,
            annotation_text=text or f"Date {x}",
            annotation_position="top right",
        )
        return self

    def add_rectangle(
        self,
        x0,
        y0,
        x1,
        y1,
        fill_color: str = "rgba(0,100,80,0.2)",
        line_color: str = "rgba(0,100,80,0.8)",
    ):
        """Add rectangle."""
        self.chart.fig.add_shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            fillcolor=fill_color,
            line=dict(color=line_color),
        )
        return self

    def clear_annotations(self):
        """Clear all annotations."""
        self.annotations.clear()
        self.chart.annotations.clear()
        return self


class RealTimeChartUpdater:
    """Real-time chart updater for streaming data."""

    def __init__(self, chart: TechnicalChart, update_interval: int = 1000):
        self.chart = chart
        self.update_interval = update_interval  # milliseconds
        self.is_running = False
        self.data_buffer = []
        self.callbacks = []

    def add_data_callback(self, callback):
        """Add callback for new data."""
        self.callbacks.append(callback)

    def update_data(self, new_data: pd.DataFrame):
        """Update chart with new data."""
        if self.chart.data is None:
            self.chart.set_data(new_data)
        else:
            # Append new data
            self.chart.data = pd.concat([self.chart.data, new_data])

            # Keep only last N points for performance
            max_points = 1000
            if len(self.chart.data) > max_points:
                self.chart.data = self.chart.data.tail(max_points)

        # Trigger callbacks
        for callback in self.callbacks:
            callback(new_data)

    def create_streaming_chart(self) -> dict:
        """Create chart configuration for streaming updates."""
        if not DASH_AVAILABLE:
            logger.warning("Dash not available for streaming charts")
            return {}

        return {
            "data": [],
            "layout": {
                "title": "Real-Time Chart",
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "Price"},
                "template": "plotly_white",
            },
        }


class AdvancedDashboard(InteractiveDashboard):
    """Advanced dashboard with additional features."""

    def __init__(self, title: str = "Advanced Technical Analysis Dashboard"):
        super().__init__(title)
        self.real_time_data = {}
        self.alerts = []
        self.watchlist = []

        # Override layout with advanced features
        self._setup_advanced_layout()
        self._setup_advanced_callbacks()

    def _setup_advanced_layout(self):
        """Setup advanced dashboard layout."""
        self.app.layout = dbc.Container(
            [
                # Header
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H1(self.title, className="text-center mb-4"),
                                html.Hr(),
                            ],
                            width=12,
                        )
                    ]
                ),
                # Main content
                dbc.Row(
                    [
                        # Left sidebar
                        dbc.Col(
                            [
                                # Symbol selection
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Symbol Selection"),
                                        dbc.CardBody(
                                            [
                                                dbc.InputGroup(
                                                    [
                                                        dbc.Input(
                                                            id="symbol-input",
                                                            placeholder="Enter symbol (e.g., AAPL)",
                                                            type="text",
                                                        ),
                                                        dbc.Button(
                                                            "Add",
                                                            id="add-symbol-button",
                                                            color="primary",
                                                            n_clicks=0,
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                html.Div(id="watchlist-container"),
                                            ]
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                # Chart controls
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Chart Controls"),
                                        dbc.CardBody(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Chart Type"),
                                                                dcc.Dropdown(
                                                                    id="chart-type-dropdown",
                                                                    options=[
                                                                        {
                                                                            "label": "Candlestick",
                                                                            "value": "candlestick",
                                                                        },
                                                                        {
                                                                            "label": "OHLC",
                                                                            "value": "ohlc",
                                                                        },
                                                                        {
                                                                            "label": "Line",
                                                                            "value": "line",
                                                                        },
                                                                        {
                                                                            "label": "Area",
                                                                            "value": "area",
                                                                        },
                                                                    ],
                                                                    value="candlestick",
                                                                ),
                                                            ],
                                                            width=12,
                                                        )
                                                    ],
                                                    className="mb-3",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Time Range"),
                                                                dcc.Dropdown(
                                                                    id="timerange-dropdown",
                                                                    options=[
                                                                        {
                                                                            "label": "1 Day",
                                                                            "value": "1d",
                                                                        },
                                                                        {
                                                                            "label": "5 Days",
                                                                            "value": "5d",
                                                                        },
                                                                        {
                                                                            "label": "1 Month",
                                                                            "value": "1mo",
                                                                        },
                                                                        {
                                                                            "label": "3 Months",
                                                                            "value": "3mo",
                                                                        },
                                                                        {
                                                                            "label": "6 Months",
                                                                            "value": "6mo",
                                                                        },
                                                                        {
                                                                            "label": "1 Year",
                                                                            "value": "1y",
                                                                        },
                                                                        {
                                                                            "label": "2 Years",
                                                                            "value": "2y",
                                                                        },
                                                                    ],
                                                                    value="3mo",
                                                                ),
                                                            ],
                                                            width=12,
                                                        )
                                                    ],
                                                    className="mb-3",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Label(
                                                                    "Technical Indicators"
                                                                ),
                                                                dcc.Checklist(
                                                                    id="indicators-checklist",
                                                                    options=[
                                                                        {
                                                                            "label": "SMA 20",
                                                                            "value": "sma20",
                                                                        },
                                                                        {
                                                                            "label": "SMA 50",
                                                                            "value": "sma50",
                                                                        },
                                                                        {
                                                                            "label": "EMA 12",
                                                                            "value": "ema12",
                                                                        },
                                                                        {
                                                                            "label": "EMA 26",
                                                                            "value": "ema26",
                                                                        },
                                                                        {
                                                                            "label": "Bollinger Bands",
                                                                            "value": "bb",
                                                                        },
                                                                        {
                                                                            "label": "RSI",
                                                                            "value": "rsi",
                                                                        },
                                                                        {
                                                                            "label": "MACD",
                                                                            "value": "macd",
                                                                        },
                                                                        {
                                                                            "label": "Stochastic",
                                                                            "value": "stoch",
                                                                        },
                                                                        {
                                                                            "label": "Volume",
                                                                            "value": "volume",
                                                                        },
                                                                    ],
                                                                    value=[
                                                                        "sma20",
                                                                        "volume",
                                                                    ],
                                                                    style={
                                                                        "fontSize": "12px"
                                                                    },
                                                                ),
                                                            ],
                                                            width=12,
                                                        )
                                                    ],
                                                    className="mb-3",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Switch(
                                                                    id="realtime-switch",
                                                                    label="Real-time Updates",
                                                                    value=False,
                                                                )
                                                            ],
                                                            width=12,
                                                        )
                                                    ],
                                                    className="mb-3",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                dbc.Button(
                                                                    "Update Chart",
                                                                    id="update-button",
                                                                    color="primary",
                                                                    className="w-100",
                                                                )
                                                            ],
                                                            width=12,
                                                        )
                                                    ]
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                # Alerts
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Price Alerts"),
                                        dbc.CardBody(
                                            [
                                                dbc.InputGroup(
                                                    [
                                                        dbc.Input(
                                                            id="alert-price-input",
                                                            placeholder="Alert price",
                                                            type="number",
                                                        ),
                                                        dbc.Button(
                                                            "Add Alert",
                                                            id="add-alert-button",
                                                            color="warning",
                                                            n_clicks=0,
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                html.Div(id="alerts-container"),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            width=3,
                        ),
                        # Main chart area
                        dbc.Col(
                            [
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="main-chart",
                                                    style={"height": "500px"},
                                                    config={
                                                        "displayModeBar": True,
                                                        "displaylogo": False,
                                                        "modeBarButtonsToAdd": [
                                                            "drawline",
                                                            "drawopenpath",
                                                            "drawclosedpath",
                                                            "drawcircle",
                                                            "drawrect",
                                                            "eraseshape",
                                                        ],
                                                    },
                                                )
                                            ]
                                        )
                                    ],
                                    className="mb-3",
                                ),
                                # Indicator charts
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            [
                                                dcc.Graph(
                                                    id="indicator-chart",
                                                    style={"height": "250px"},
                                                )
                                            ]
                                        )
                                    ]
                                ),
                            ],
                            width=9,
                        ),
                    ]
                ),
                # Real-time updates interval
                dcc.Interval(
                    id="interval-component",
                    interval=5 * 1000,  # Update every 5 seconds
                    n_intervals=0,
                    disabled=True,
                ),
                # Store for data
                dcc.Store(id="chart-data-store"),
                dcc.Store(id="selected-symbol-store"),
            ],
            fluid=True,
        )

    def _setup_advanced_callbacks(self):
        """Setup advanced dashboard callbacks."""

        @self.app.callback(
            Output("watchlist-container", "children"),
            Output("selected-symbol-store", "data"),
            [Input("add-symbol-button", "n_clicks")],
            [State("symbol-input", "value"), State("selected-symbol-store", "data")],
        )
        def update_watchlist(n_clicks, symbol, current_selection):
            if n_clicks > 0 and symbol:
                if symbol not in self.watchlist:
                    self.watchlist.append(symbol.upper())

            # Create watchlist display
            watchlist_items = []
            for sym in self.watchlist:
                watchlist_items.append(
                    dbc.ListGroupItem(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(sym, width=8),
                                    dbc.Col(
                                        [
                                            dbc.Button(
                                                "Select",
                                                size="sm",
                                                color="outline-primary",
                                                id=f"select-{sym}",
                                                n_clicks=0,
                                            )
                                        ],
                                        width=4,
                                    ),
                                ]
                            )
                        ]
                    )
                )

            return dbc.ListGroup(watchlist_items), current_selection

        @self.app.callback(
            Output("interval-component", "disabled"),
            [Input("realtime-switch", "value")],
        )
        def toggle_realtime(realtime_enabled):
            return not realtime_enabled

        @self.app.callback(
            [Output("main-chart", "figure"), Output("indicator-chart", "figure")],
            [
                Input("update-button", "n_clicks"),
                Input("interval-component", "n_intervals"),
            ],
            [
                State("selected-symbol-store", "data"),
                State("chart-type-dropdown", "value"),
                State("timerange-dropdown", "value"),
                State("indicators-checklist", "value"),
                State("realtime-switch", "value"),
            ],
        )
        def update_charts(
            update_clicks,
            interval_clicks,
            selected_symbol,
            chart_type,
            timerange,
            indicators,
            realtime,
        ):
            if not selected_symbol or selected_symbol not in self.data_sources:
                return self._create_empty_chart(), self._create_empty_chart()

            data = self.data_sources[selected_symbol]

            # Filter data by time range
            if timerange != "all":
                end_date = data.index[-1]
                if timerange == "1d":
                    start_date = end_date - timedelta(days=1)
                elif timerange == "5d":
                    start_date = end_date - timedelta(days=5)
                elif timerange == "1mo":
                    start_date = end_date - timedelta(days=30)
                elif timerange == "3mo":
                    start_date = end_date - timedelta(days=90)
                elif timerange == "6mo":
                    start_date = end_date - timedelta(days=180)
                elif timerange == "1y":
                    start_date = end_date - timedelta(days=365)
                elif timerange == "2y":
                    start_date = end_date - timedelta(days=730)

                data = data[data.index >= start_date]

            # Create charts
            main_fig = self._create_advanced_main_chart(data, chart_type, indicators)
            indicator_fig = self._create_advanced_indicator_chart(data, indicators)

            return main_fig, indicator_fig

    def _create_empty_chart(self):
        """Create empty chart placeholder."""
        return {
            "data": [],
            "layout": {
                "title": "No data available",
                "template": "plotly_white",
                "annotations": [
                    {
                        "text": "Select a symbol to view chart",
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.5,
                        "y": 0.5,
                        "xanchor": "center",
                        "yanchor": "middle",
                        "showarrow": False,
                        "font": {"size": 16},
                    }
                ],
            },
        }

    def _create_advanced_main_chart(
        self, data: pd.DataFrame, chart_type: str, indicators: List[str]
    ):
        """Create advanced main chart with multiple indicators."""
        # Determine if we need subplots
        has_volume = "volume" in indicators
        subplot_count = 1 + (1 if has_volume else 0)

        if subplot_count > 1:
            fig = make_subplots(
                rows=subplot_count,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.7, 0.3] if has_volume else [1.0],
                subplot_titles=["Price", "Volume"] if has_volume else ["Price"],
            )
        else:
            fig = go.Figure()

        # Add price data
        if chart_type == "candlestick":
            candlestick = go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Price",
                increasing_line_color="#00ff00",
                decreasing_line_color="#ff0000",
            )

            if subplot_count > 1:
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)

        elif chart_type == "ohlc":
            ohlc = go.Ohlc(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Price",
            )

            if subplot_count > 1:
                fig.add_trace(ohlc, row=1, col=1)
            else:
                fig.add_trace(ohlc)

        elif chart_type == "area":
            area = go.Scatter(
                x=data.index,
                y=data["Close"],
                mode="lines",
                name="Close Price",
                fill="tonexty",
                fillcolor="rgba(0,100,80,0.2)",
            )

            if subplot_count > 1:
                fig.add_trace(area, row=1, col=1)
            else:
                fig.add_trace(area)
        else:  # line
            line = go.Scatter(
                x=data.index,
                y=data["Close"],
                mode="lines",
                name="Close Price",
                line=dict(color="blue"),
            )

            if subplot_count > 1:
                fig.add_trace(line, row=1, col=1)
            else:
                fig.add_trace(line)

        # Add technical indicators to main chart
        row_num = 1

        if "sma20" in indicators:
            sma20 = data["Close"].rolling(window=20).mean()
            sma_trace = go.Scatter(
                x=data.index,
                y=sma20,
                mode="lines",
                name="SMA 20",
                line=dict(color="blue", width=1),
            )

            if subplot_count > 1:
                fig.add_trace(sma_trace, row=row_num, col=1)
            else:
                fig.add_trace(sma_trace)

        if "sma50" in indicators:
            sma50 = data["Close"].rolling(window=50).mean()
            sma_trace = go.Scatter(
                x=data.index,
                y=sma50,
                mode="lines",
                name="SMA 50",
                line=dict(color="red", width=1),
            )

            if subplot_count > 1:
                fig.add_trace(sma_trace, row=row_num, col=1)
            else:
                fig.add_trace(sma_trace)

        if "ema12" in indicators:
            ema12 = data["Close"].ewm(span=12).mean()
            ema_trace = go.Scatter(
                x=data.index,
                y=ema12,
                mode="lines",
                name="EMA 12",
                line=dict(color="green", width=1),
            )

            if subplot_count > 1:
                fig.add_trace(ema_trace, row=row_num, col=1)
            else:
                fig.add_trace(ema_trace)

        if "ema26" in indicators:
            ema26 = data["Close"].ewm(span=26).mean()
            ema_trace = go.Scatter(
                x=data.index,
                y=ema26,
                mode="lines",
                name="EMA 26",
                line=dict(color="orange", width=1),
            )

            if subplot_count > 1:
                fig.add_trace(ema_trace, row=row_num, col=1)
            else:
                fig.add_trace(ema_trace)

        if "bb" in indicators:
            sma = data["Close"].rolling(window=20).mean()
            std = data["Close"].rolling(window=20).std()

            bb_upper = go.Scatter(
                x=data.index,
                y=sma + (std * 2),
                mode="lines",
                name="BB Upper",
                line=dict(color="rgba(0,100,80,0)", width=0),
                showlegend=False,
            )

            bb_lower = go.Scatter(
                x=data.index,
                y=sma - (std * 2),
                mode="lines",
                name="Bollinger Bands",
                line=dict(color="rgba(0,100,80,0)", width=0),
                fill="tonexty",
                fillcolor="rgba(0,100,80,0.2)",
            )

            if subplot_count > 1:
                fig.add_trace(bb_upper, row=row_num, col=1)
                fig.add_trace(bb_lower, row=row_num, col=1)
            else:
                fig.add_trace(bb_upper)
                fig.add_trace(bb_lower)

        # Add volume if requested
        if has_volume and "Volume" in data.columns:
            volume_trace = go.Bar(
                x=data.index,
                y=data["Volume"],
                name="Volume",
                marker_color="rgba(0,100,80,0.5)",
                yaxis="y2" if subplot_count == 1 else None,
            )

            if subplot_count > 1:
                fig.add_trace(volume_trace, row=2, col=1)
            else:
                fig.add_trace(volume_trace)

        # Update layout
        fig.update_layout(
            title="Technical Analysis Chart",
            template="plotly_white",
            height=500,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        if subplot_count == 1 and has_volume:
            # Add secondary y-axis for volume
            fig.update_layout(
                yaxis2=dict(
                    title="Volume", overlaying="y", side="right", showgrid=False
                )
            )

        return fig

    def _create_advanced_indicator_chart(
        self, data: pd.DataFrame, indicators: List[str]
    ):
        """Create advanced indicator chart."""
        indicator_traces = []

        if "rsi" in indicators:
            # Calculate RSI
            delta = data["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            indicator_traces.append(
                {
                    "name": "RSI",
                    "data": rsi,
                    "color": "purple",
                    "type": "line",
                    "reference_lines": [30, 50, 70],
                }
            )

        if "macd" in indicators:
            # Calculate MACD
            ema12 = data["Close"].ewm(span=12).mean()
            ema26 = data["Close"].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal

            indicator_traces.extend(
                [
                    {"name": "MACD", "data": macd, "color": "blue", "type": "line"},
                    {"name": "Signal", "data": signal, "color": "red", "type": "line"},
                    {
                        "name": "Histogram",
                        "data": histogram,
                        "color": "gray",
                        "type": "bar",
                    },
                ]
            )

        if "stoch" in indicators:
            # Calculate Stochastic
            low_14 = data["Low"].rolling(window=14).min()
            high_14 = data["High"].rolling(window=14).max()
            k_percent = 100 * ((data["Close"] - low_14) / (high_14 - low_14))
            d_percent = k_percent.rolling(window=3).mean()

            indicator_traces.extend(
                [
                    {
                        "name": "%K",
                        "data": k_percent,
                        "color": "blue",
                        "type": "line",
                        "reference_lines": [20, 50, 80],
                    },
                    {"name": "%D", "data": d_percent, "color": "red", "type": "line"},
                ]
            )

        # Create figure
        fig = go.Figure()

        for trace_info in indicator_traces:
            if trace_info["type"] == "line":
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=trace_info["data"],
                        mode="lines",
                        name=trace_info["name"],
                        line=dict(color=trace_info["color"]),
                    )
                )
            elif trace_info["type"] == "bar":
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=trace_info["data"],
                        name=trace_info["name"],
                        marker_color=trace_info["color"],
                    )
                )

            # Add reference lines if specified
            if "reference_lines" in trace_info:
                for level in trace_info["reference_lines"]:
                    line_color = (
                        "red"
                        if level in [70, 80]
                        else "green"
                        if level in [20, 30]
                        else "gray"
                    )
                    fig.add_hline(
                        y=level, line_dash="dash", line_color=line_color, opacity=0.5
                    )

        fig.update_layout(
            title="Technical Indicators",
            template="plotly_white",
            height=250,
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        return fig

    def add_real_time_data(self, symbol: str, new_data: pd.DataFrame):
        """Add real-time data update."""
        if symbol in self.data_sources:
            # Append new data
            self.data_sources[symbol] = pd.concat([self.data_sources[symbol], new_data])

            # Keep only recent data for performance
            max_points = 5000
            if len(self.data_sources[symbol]) > max_points:
                self.data_sources[symbol] = self.data_sources[symbol].tail(max_points)
        else:
            self.data_sources[symbol] = new_data

    def add_price_alert(self, symbol: str, price: float, condition: str = "above"):
        """Add price alert."""
        alert = {
            "symbol": symbol,
            "price": price,
            "condition": condition,
            "created": datetime.now(),
            "triggered": False,
        }

        self.alerts.append(alert)
        logger.info(f"Added price alert for {symbol}: {condition} {price}")

    def check_alerts(self, symbol: str, current_price: float):
        """Check if any alerts should be triggered."""
        triggered_alerts = []

        for alert in self.alerts:
            if alert["symbol"] == symbol and not alert["triggered"]:
                if (
                    alert["condition"] == "above" and current_price >= alert["price"]
                ) or (
                    alert["condition"] == "below" and current_price <= alert["price"]
                ):
                    alert["triggered"] = True
                    triggered_alerts.append(alert)

        return triggered_alerts
