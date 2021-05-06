from dataclasses import dataclass
import typing
from typing import Iterable
import numpy as np

FigureType = typing.NewType('FigureType', str)
TablePrice = FigureType("TablePrice")
TableDuration = FigureType("TableDuration")
PricePerNbPaths = FigureType("PricePerNbPaths")


@dataclass
class _DefaultConfig:
  algos: Iterable[str] = ('NLSM',
                          'LSM',
                          'DOS',
                          'FQI',
                          'RFQI',
                          'RLSM')
  dividends: Iterable[float] = (0.0,)
  nb_dates: Iterable[int] = (10,)
  drift: Iterable[float] = (0.02,)
  mean: Iterable[float] = (0.01,)
  speed: Iterable[float] = (2,)
  correlation: Iterable[float] = (-0.3,)
  hurst: Iterable[float] = (0.75,)
  stock_models: Iterable[str] = ('BlackScholes',)
  strikes: Iterable[int] = (100,)
  maturities: Iterable[float] = (1,)
  nb_paths: Iterable[int] = (20000,)
  nb_runs: int = 10
  nb_stocks: Iterable[int] = (1,)
  payoffs: Iterable[str] = ('MaxCall',)
  spots: Iterable[int] = (100,)
  volatilities: Iterable[float] = (0.2,)
  hidden_size: Iterable[int] = (20,)
  nb_epochs: Iterable[int] = (30,)
  factors: Iterable[Iterable[float]] = ((1.,1.,1.),)
  ridge_coeff: Iterable[float] = (1.,)
  train_ITM_only: Iterable[bool] = (True,)
  use_path: Iterable[bool] = (False,)
  representations: Iterable[str] = ('TablePriceDuration',)
  # When adding a filter here, also add to filtering.py.


'''
Comparison prices and computation time
'''

@dataclass
class _DimensionTable(_DefaultConfig):
  algos: Iterable[str] = ('NLSM', 'RFQI', 'RLSM')
  nb_stocks: Iterable[int] = (5, 10, 50, 100, 500, 1000, 2000)


table_spots_Dim_BS_MaxCall = _DimensionTable(spots=[80, 100, 120])
table_Dim_Heston_MaxCall = _DimensionTable(stock_models=['Heston'])
table_Dim_FracBS_MaxCall = _DimensionTable(
    stock_models = ['FractionalBlackScholes'])

algos = ['DOS',]
table_spots_Dim_BS_MaxCall_do = _DimensionTable(
    algos=algos,
    spots=[80, 100,  120])
table_Dim_Heston_MaxCall_do = _DimensionTable(
    algos=algos,
    stock_models=['Heston'])
table_Dim_FracBS_MaxCall_do = _DimensionTable(
    algos=algos,
    stock_models=['FractionalBlackScholes'])


# tables with basis functions
@dataclass
class _SmallDimensionTable(_DefaultConfig):
  algos: Iterable[str] = ('LSM', 'FQI')
  nb_stocks: Iterable[int] = (5, 10, 50, 100)


table_spots_Dim_BS_MaxCall_bf = _SmallDimensionTable(spots=[80, 100,  120])
table_Dim_Heston_MaxCall_bf = _SmallDimensionTable(stock_models=['Heston'])
table_Dim_FracBS_MaxCall_bf = _SmallDimensionTable(
    stock_models=['FractionalBlackScholes'])

# tables to generate output tables
algos = ['NLSM', 'RFQI', 'RLSM', 'LSM', 'FQI', 'DOS']
table_spots_Dim_BS_MaxCall_gt = _DimensionTable(
    spots=[80, 100,  120], algos=algos)
table_Dim_Heston_MaxCall_gt = _DimensionTable(
    stock_models=['Heston'], algos=algos)
table_Dim_FracBS_MaxCall_gt = _DimensionTable(
    stock_models=['FractionalBlackScholes'], algos=algos)


@dataclass
class _VerySmallDimensionTable(_DefaultConfig):
  nb_stocks: Iterable[int] = (1, 5, 10, 20)


table_smallDim_BS_GeoPut_BasketCall = _VerySmallDimensionTable(
    payoffs=['GeometricPut', 'BasketCall'], algos=algos)

table_smallDim_BS_GeoPut = _VerySmallDimensionTable(
    payoffs=['GeometricPut'], algos=algos)
table_Dim_BS_BasktCall = _DimensionTable(
    payoffs=['BasketCall'], algos=('NLSM', 'RFQI', 'RLSM', 'DOS'))
table_Dim_BS_BasktCall_bf = _SmallDimensionTable(payoffs=['BasketCall'])
table_other_payoffs_gt = _DimensionTable(
    payoffs=['BasketCall', 'GeometricPut'], algos=algos,
    nb_stocks=[1, 5, 10, 20, 50, 100, 500, 1000, 2000])

algos = ['NLSM', 'RFQI', 'RLSM', 'LSM', 'DOS']
table_manyDates_BS_MaxCall1 = _VerySmallDimensionTable(
    algos=algos, nb_stocks=[10, 50, ], nb_dates=[50, 100])
algos_ = ['NLSM', 'RFQI', 'RLSM', 'DOS']
table_manyDates_BS_MaxCall2 = _VerySmallDimensionTable(
    algos=algos_, nb_stocks=[100, 500,], nb_dates=[50, 100])
table_manyDates_BS_MaxCall3 = _VerySmallDimensionTable(
    algos=["RFQI"], nb_stocks=[10, 50, 100, 500,], nb_dates=[50, 100],
    hidden_size=[40, 80]
)
table_manyDates_BS_MaxCall_gt = _VerySmallDimensionTable(
    algos=algos, nb_stocks=[10, 50, 100, 500,], nb_dates=[50, 100],
    hidden_size=[20]
)



'''
Empirical convergence study
'''

@dataclass
class _DefaultPlotNbPaths(_DefaultConfig):
  nb_runs: int = 20
  nb_stocks = [5, ]
  maturities: Iterable[int] = (1,)
  representations: Iterable[str] = ("ConvergenceStudy",)


table_conv_study_Heston_LND = _DefaultPlotNbPaths(
    nb_paths=list(200 * 2**np.array(range(8))),
    hidden_size=(10,50,100,),
    algos=("RLSM", ),
    stock_models=['Heston'],
)
table_conv_study_BS_LND = _DefaultPlotNbPaths(
    nb_paths=list(200 * 2**np.array(range(8))),
    hidden_size=(10,50,100,),
    algos=("RLSM", ),
    stock_models=['BlackScholes'],
)
table_conv_study_Heston_FQIR = _DefaultPlotNbPaths(
    nb_paths=list(200 * 2**np.array(range(8))),
    hidden_size=(5, 10, 50, 100),
    algos=("RFQI",),
    stock_models=['Heston'],
)
table_conv_study_BS_FQIR = _DefaultPlotNbPaths(
    nb_paths=list(200 * 2**np.array(range(8))),
    hidden_size=(5, 10, 50, 100),
    algos=("RFQI",),
    stock_models=['BlackScholes'],
)


'''
Test for the FBM case of DOS
'''
# RNNLeastSquares
hurst = list(np.linspace(0, 1, 21))
hurst[0] = 0.01
hurst[-1] = 0.999
# hurst = hurst[0:3]
# hurst = hurst[-2:]
# hurst = [0.5]
table_RNN_DOS = _DefaultConfig(
    payoffs=['Identity'], nb_stocks=[1], spots=[0], nb_epochs=[30],
    hurst=hurst, train_ITM_only=[False],
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'NLSM',
        'DOS',
        'RLSM',
        'RFQI',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)
table_RNN_DOS_PD = _DefaultConfig(
    payoffs=['Identity'], nb_stocks=[1], spots=[0], nb_epochs=[30],
    hurst=hurst, train_ITM_only=[False],
    stock_models=['FractionalBrownianMotionPathDep'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'NLSM',
        'DOS',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)
table_RNN_DOS_bf = _DefaultConfig(
    payoffs=['Identity'], nb_stocks=[1], spots=[0], nb_epochs=[30],
    hurst=hurst, train_ITM_only=[False],
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'LSM',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)

factors0 = []
for a in [0.0001]:
    for b in [0.3]:
        factors0 += [[a,b]]
table_RNN_DOS_randRNN = _DefaultConfig(
    payoffs=['Identity'], nb_stocks=[1], spots=[0], nb_epochs=[30],
    hurst=hurst, train_ITM_only=[False],
    factors=factors0,
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'RRLSM',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)
table_RNN_DOS_FQIR_PD = _DefaultConfig(
    payoffs=['Identity'], nb_stocks=[1], spots=[0], nb_epochs=[30],
    hurst=hurst, train_ITM_only=[False],
    stock_models=['FractionalBrownianMotionPathDep'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'RFQI',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)
table_RNN_DOS_FQIRRNN = _DefaultConfig(
    payoffs=['Identity'], nb_stocks=[1], spots=[0], nb_epochs=[30],
    hurst=hurst, train_ITM_only=[False],
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'RRFQI'
    ], nb_runs=10,
    representations=['TablePriceDuration']
)




'''
higher dimensional FBM
'''
table_highdim_hurst0 = _DefaultConfig(
    payoffs=['Identity'],
    nb_stocks=[1],
    spots=[0], nb_epochs=[30],
    hurst=[0.05], train_ITM_only=[False],
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'DOS',
        'RLSM',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)
table_highdim_hurst_PD0 = _DefaultConfig(
    payoffs=['Identity'],
    nb_stocks=[1],
    spots=[0], nb_epochs=[30],
    hurst=[0.05], train_ITM_only=[False],
    use_path=[True],
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'DOS',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)
factors = []
for a in [0.0008,]:
    for b in [0.11,]:
        factors += [[a,b]]
table_highdim_hurst_RNN0 = _DefaultConfig(
    payoffs=['Identity'],
    nb_stocks=[1],
    spots=[0], nb_epochs=[30],
    hurst=[0.05], train_ITM_only=[False],
    factors=factors,
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'RRLSM',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)

hurst1 = [0.05]
table_highdim_hurst = _DefaultConfig(
    payoffs=['Max', 'Mean'],
    nb_stocks=[5, 10, 50, 100,],
    spots=[0], nb_epochs=[30],
    hurst=hurst1, train_ITM_only=[False],
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'DOS',
        'RLSM',
        'RFQI',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)
table_highdim_hurst_PD = _DefaultConfig(
    payoffs=['Max', 'Mean'],
    nb_stocks=[5, 10,],
    spots=[0], nb_epochs=[30],
    hurst=hurst1, train_ITM_only=[False],
    use_path=[True],
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'DOS',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)
factors = []
for a in [0.0008,]:
    for b in [0.11,]:
        factors += [[a,b]]
table_highdim_hurst_RNN = _DefaultConfig(
    payoffs=['Max', 'Mean'],
    nb_stocks=[5, 10,],
    spots=[0], nb_epochs=[30],
    hurst=hurst1, train_ITM_only=[False],
    factors=factors,
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'RRLSM',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)

factors1 = [(1.,1.,1.), [0.0008,0.11]]
table_highdim_hurst_gt = _DefaultConfig(
    payoffs=['Identity', 'Max', 'Mean'],
    nb_stocks=[1, 5, 10, ],
    spots=[0], nb_epochs=[30],
    hurst=[0.05,], train_ITM_only=[False],
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100], factors=factors1,
    algos=[
        'DOS',
        'pathDO',
        'RLSM',
        'RRLSM',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)


'''
test with Ridge regression
'''
table_Ridge_MaxCall = _SmallDimensionTable(
    spots=[100], algos=["LSMRidge", "RLSMRidge"], ridge_coeff=[1., 0.5, 2.],
    representations=['TablePriceDuration'],
)


'''
test with other basis functions
'''
table_OtherBasis_MaxCall = _SmallDimensionTable(
    spots=[100], algos=[
        "LSMLaguerre", "LSM", "FQILaguerre", "FQI", ],
    nb_stocks=(5, 10, 50,),
    nb_runs=10,
    representations=['TablePriceDuration'],
)





test_table = _SmallDimensionTable(
    spots=[10], strikes=[10],
    algos=[
        'NLSM', 'LSM', 'DOS', 'FQI', 'RFQI', 'RLSM',
        "LSMLaguerre", "FQILaguerre", "LSMRidge", "RLSMRidge",
        "RRLSM", "RRFQI", "LSPI",

    ],
    nb_stocks=(5,), nb_dates=(5,), nb_paths=(100,),
    nb_runs=2, factors=((0.001,0.001,0.001),),
    representations=['TablePriceDuration'],
)