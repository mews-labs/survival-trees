import matplotlib.pyplot as plot
import numpy as np
import pandas as pd

default_event_observed_color = "#A60628"
default_event_censored_color = "#348ABD"
default_event_observed_color_2 = "red"
default_event_censored_color_2 = "blue"
default_plotly_template = "plotly_white"


def tagged_curves(temporal_curves: pd.DataFrame, label: pd.DataFrame,
                  time_event: pd.Series = None,
                  event_observed_color="#A60628",
                  event_censored_color="#348ABD",
                  add_marker=False
                  ):
    temp_curves = temporal_curves.astype("float16")
    temp_curves.index = range(len(temp_curves))
    dates = temp_curves.columns.to_list()
    pos_index = temp_curves.index[label.astype(bool)]
    neg_index = temp_curves.index[~label.astype(bool)]
    # check consistency
    if time_event is not None:
        col_id = np.searchsorted(temp_curves.columns, time_event, side="right") - 1
        col = temp_curves.columns[col_id]
        if add_marker:
            prob = np.diag(temp_curves[col])
            plot.scatter(col[label.astype(bool)], prob[label.astype(bool)],
                         marker="*",
                         c=event_observed_color)
        outdated = pd.DataFrame(np.nan, index=temp_curves.index,
                                columns=temp_curves.columns)
        for i, ind in enumerate(temp_curves.index):
            data = temp_curves.loc[ind]
            outdated.loc[ind, dates[col_id[i]:]] = data[dates[col_id[i]:]]
            temp_curves.loc[ind, dates[col_id[i] + 1:]] = np.nan
        plot_args = dict(alpha=0.4, ls="--", lw=1)
        plot.plot(outdated.loc[neg_index].T, c=event_censored_color,
                  **plot_args)
        plot.plot(outdated.loc[pos_index].T, c=event_observed_color,
                  **plot_args)
    plot_args = dict(lw=1, alpha=0.9)
    plot.plot(temp_curves.loc[pos_index].T, c=event_observed_color, **plot_args)
    plot.plot(temp_curves.loc[neg_index].T, c=event_censored_color, **plot_args)


def _prepare_data(duration,
                  event,
                  entry,
                  temporal_score: pd.DataFrame
                  ):
    label = event
    time_event = duration
    temp_curves = temporal_score.astype("float32")
    temp_curves.index = range(len(temp_curves))
    dates = temp_curves.columns.to_list()
    pos_index = temp_curves.index[label.astype(bool)]
    # check consistency
    if time_event is not None:
        col_id = np.searchsorted(temp_curves.columns, time_event) - 1

        outdated = pd.DataFrame(np.nan, index=temp_curves.index,
                                columns=temp_curves.columns)
        for c in np.unique(col_id):
            ind = col_id == c
            data = temp_curves[ind]
            outdated.loc[ind, dates[c:]] = data[dates[c:]]
            temp_curves.loc[ind, dates[c + 1:]] = np.nan
        temp_curves["observed"] = True
        outdated["observed"] = False
        data = pd.concat((temp_curves, outdated), axis=0)

        data["observed event"] = False
        data.loc[pos_index, "observed event"] = True
        return data


def auc_vs_score_plotly(
        temporal_scores: pd.DataFrame,
        event_observed: pd.Series,
        censoring_time: pd.Series,
        entry_time: pd.Series,
        temporal_metric: pd.Series,
        event_observed_color=default_event_observed_color_2,
        event_censored_color=default_event_censored_color_2,
        template=default_plotly_template,
        colormap="magma_r",
        plot_random_prediction=False):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    data = _prepare_data(duration=censoring_time,
                         event=event_observed,
                         entry=entry_time,
                         temporal_score=temporal_scores)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[1000, 550])
    t = True
    for d in np.unique(data.index):
        data_t = data.loc[d]
        u = + np.random.rand() * 0.05
        color = event_observed_color if data_t.loc[d]["observed event"].iloc[0] else event_censored_color
        showlegend = bool(data_t.loc[d]["observed event"].iloc[0] and t)
        t = not t if showlegend else t
        fig.add_traces(
            [go.Scatter(
                x=data.columns.tolist(),
                y=data.loc[d].iloc[0] + u,
                mode='lines',
                name="Censored event",
                line=dict(color=event_censored_color, width=1),
                showlegend=showlegend,
            )])
        fig.add_traces(
            [go.Scatter(
                x=data.columns.tolist(),
                y=data.loc[d].iloc[1] + u,
                line=dict(color=color,
                          width=0.9,
                          # dash='dash'
                          ),
                mode='lines',
                name="Observed event",
                showlegend=showlegend
            )])
    for i in range(temporal_metric.__len__() - 1):
        tmp_legend = i <= 0
        fig.add_traces(
            [go.Scatter(x=[temporal_metric.index[i], temporal_metric.index[i + 1]],
                        y=[temporal_metric.iloc[i], temporal_metric.iloc[i + 1]],
                        showlegend=tmp_legend, name="Time dependent AUC", line=dict(color="black", width=1),
                        mode="lines")],
            rows=2,
            cols=1,
        )
    color = (temporal_metric.index - temporal_metric.index.min()) / (
            temporal_metric.index.max() - temporal_metric.index.min())

    fig.add_traces(
        [go.Scatter(x=[temporal_metric.index.tolist()[-1]], y=[temporal_metric.values[-1]],
                    mode="markers",
                    marker={'color': "black", 'size': 10},
                    name="Time dependent AUC"
                    )],
        rows=2,
        cols=1
    )

    fig.add_traces(
        [go.Scatter(x=temporal_metric.index.tolist(), y=temporal_metric.values,
                    mode="markers",
                    marker={'color': 1 - color, 'colorscale': colormap, 'size': 10},
                    name="Time dependent AUC", showlegend=False,
                    )],
        rows=2,
        cols=1
    )

    if plot_random_prediction:
        fig.add_traces(
            [go.Scatter(x=[temporal_metric.index[0], temporal_metric.index[-1]], y=[0.5, 0.5],
                        line=dict(color="black", width=2, dash='dash'),
                        name="Random prediction")],
            rows=2,
            cols=1)
    fig.update_layout(
        width=750,
        margin=dict(
            l=1,
            r=2,
            b=1,
            t=1,
        ))
    fig.update_yaxes(title_text="Survival curves", row=1)
    fig.update_yaxes(title_text="AUC", row=2)
    fig.update_xaxes(title_text="Time", row=2)
    fig.update_layout(template=template)
    return fig


def plot_roc_curve_plotly(
        roc: dict,
        colormap: str = "magma_r",
        template: str = default_plotly_template,
        marker_size: int = 10,
):
    import plotly.graph_objects as go
    import matplotlib
    max_, min_ = max(roc.keys()), min(roc.keys())
    colormap_ = plot.get_cmap(colormap)
    fig = go.Figure()
    for i, (t, (fpr, tpr, th)) in enumerate(roc.items()):
        c = matplotlib.colors.rgb2hex(colormap_(1 - (t - min_) / (max_ - min_)))

        trace_line = go.Scatter(
            x=fpr, y=tpr,
            line={"color": c},
            mode="lines+markers",
            marker={'color': c, 'size': marker_size},
            name=f"ROC(t={t})"
        )
        trace_line["showlegend"] = (i % 4) == 0
        fig.add_traces([trace_line])
        # fig.add_traces(trace_sc)

    trace_line = go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(color="black", width=1, dash='dash'),
        mode="lines",
        name=f"random prediction"
    )
    fig.add_traces([trace_line])

    fig.update_layout(
        width=740,
        height=700,
        showlegend=True,
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        margin=dict(
            l=1,
            r=2,
            b=1,
            t=1,
        ))
    fig.update_layout(template=template)
    return fig
