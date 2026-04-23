

if __name__ == '__main__':
    from lifelines import (WeibullFitter, ExponentialFitter,
                           LogNormalFitter, LogLogisticFitter, NelsonAalenFitter,
                           PiecewiseExponentialFitter, GeneralizedGammaFitter, SplineFitter)

    from lifelines.datasets import load_waltons
    import matplotlib.pyplot as plt
    data = load_waltons()

    fig, axes = plt.subplots(3, 3, figsize=(10, 7.5))

    T = data['T']
    E = data['E']

    wbf = WeibullFitter().fit(T, E, label='WeibullFitter')
    exf = ExponentialFitter().fit(T, E, label='ExponentialFitter')
    lnf = LogNormalFitter().fit(T, E, label='LogNormalFitter')
    naf = NelsonAalenFitter().fit(T, E, label='NelsonAalenFitter')
    llf = LogLogisticFitter().fit(T, E, label='LogLogisticFitter')
    pwf = PiecewiseExponentialFitter([40, 60]).fit(T, E, label='PiecewiseExponentialFitter')
    gg = GeneralizedGammaFitter().fit(T, E, label='GeneralizedGammaFitter')
    spf = SplineFitter([6, 20, 40, 75]).fit(T, E, label='SplineFitter')

    wbf.plot_survival_function(ax=axes[0][0])
    exf.plot_survival_function(ax=axes[0][1])
    lnf.plot_survival_function(ax=axes[0][2])
    naf.plot_survival_function(ax=axes[1][0])
    llf.plot_survival_function(ax=axes[1][1])
    pwf.plot_survival_function(ax=axes[1][2])
    gg.plot_survival_function(ax=axes[2][0])
    spf.plot_survival_function(ax=axes[2][1])
    axes[2][2].set_axis_off()
