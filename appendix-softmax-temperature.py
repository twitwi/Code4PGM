import marimo

__generated_with = "0.8.16"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import altair as alt
    import numpy as np
    import pandas as pd
    return alt, mo, np, pd


@app.cell
def __(np, pd):
    def softmax(logp, t=1):
        res = np.exp(logp / t)
        return res / np.sum(res)

    nclass = 30
    logp = np.random.normal(0, 1, nclass)
    logp[3:5] += 1
    logp[7] += 5

    data = pd.DataFrame(np.arange(nclass), columns=['cl'])
    data['logit'] = logp
    data['proba'] = softmax(logp)
    data
    return data, logp, nclass, softmax


@app.cell
def __(mo):
    logt = mo.ui.slider(-3, 5, step=0.1, value=0)
    logt
    return logt,


@app.cell
def __(data, logt, np, softmax):
    datamore = data.copy()
    datamore['probatemp'] = softmax(data['logit'], np.exp(logt.value))
    return datamore,


@app.cell
def __(alt, datamore, mo):
    chart = alt.Chart(datamore).mark_bar().encode(
        x='cl',
        y='probatemp'
    )
    mo.ui.altair_chart(chart)
    return chart,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
