from dataclasses import dataclass
import typing
from typing import Iterable
import numpy as np

FigureType = typing.NewType('FigureType', str)
TablePrice = FigureType("TablePrice")
TableDuration = FigureType("TableDuration")
PricePerNbPaths = FigureType("PricePerNbPaths")

'''
Algos:NLSM, LSM, LS2, DOS, HP, LSPI, FQI, FQIR, LN, LN2, LND
New algos = RLSM, FQI, RFQI, LNfast
'''

class Seed:
    def __init__(self):
        self.seed = None

    def set_seed(self, seed):
        self.seed = seed

    def get_seed(self):
        return self.seed


path_gen_seed = Seed()


@dataclass
class _DefaultConfig:
  algos: Iterable[str] = ('NLSM',
                          'LSM',
                          'DOS',
                          'FQI',
                          'RFQI',
                          'LNfast',
                          'LN2',
                          'RLSM')
  dividends: Iterable[float] = (0.0,)
  nb_dates: Iterable[int] = (10,)
  drift: Iterable[float] = (0.02,)
  mean: Iterable[float] = (0.01,)
  speed: Iterable[float] = (2,)
  correlation: Iterable[float] = (-0.3,)
  hurst: Iterable[float] = (0.75,)
  stock_models: Iterable[str] = ('BlackScholes',)
  strikes: Iterable[float] = (100,)
  maturities: Iterable[float] = (1,)
  nb_paths: Iterable[int] = (20000,)
  nb_runs: int = 10
  nb_stocks: Iterable[int] = (1,)
  payoffs: Iterable[str] = ('MaxCall',)
  spots: Iterable[float] = (100,)
  volatilities: Iterable[float] = (0.2,)
  hidden_size: Iterable[int] = (20,)
  nb_epochs: Iterable[int] = (30,)
  factors: Iterable[Iterable[float]] = ((1.,1.,1.),)
  ridge_coeff: Iterable[float] = (1.,)
  train_ITM_only: Iterable[bool] = (True,)
  use_path: Iterable[bool] = (False,)
  use_payoff_as_input: Iterable[bool] = (False,)
  representations: Iterable[str] = ('TablePriceDuration',)
  # When adding a filter here, also add to filtering.py and read_data.py


'''
Comparison prices and computation time
'''

@dataclass
class _DimensionTable(_DefaultConfig):
  algos: Iterable[str] = ('NLSM', 'RFQI', 'RLSM')
  nb_stocks: Iterable[int] = (5, 10, 50, 100, 500, 1000, 2000)


# tables with basis functions
@dataclass
class _SmallDimensionTable(_DefaultConfig):
  algos: Iterable[str] = ('LSM', 'FQI')
  nb_stocks: Iterable[int] = (5, 10, 50, 100)


@dataclass
class _VerySmallDimensionTable(_DefaultConfig):
  nb_stocks: Iterable[int] = (5, 10, 20)
  algos: Iterable[str] = ('NLSM', 'RFQI', 'RLSM', 'LSM', 'FQI', 'DOS')



'''
Comparison prices and computation time 
'''

# BS and Heston MaxCall
table_spots_Dim_BS_MaxCallr0 = _DimensionTable(
    spots=[80, 100, 120], drift=(0.0,), use_payoff_as_input=(True, False))
table_Dim_Heston_MaxCallr0 = _DimensionTable(
    stock_models=['Heston'], drift=(0.0,), use_payoff_as_input=(True, False))
#   -- table for do
algos = ['DOS',]
table_spots_Dim_BS_MaxCallr0_do = _DimensionTable(
    algos=algos, drift=(0.0,), use_payoff_as_input=(True, False),
    spots=[80, 100,  120])
table_Dim_Heston_MaxCallr0_do = _DimensionTable(
    algos=algos, drift=(0.0,), use_payoff_as_input=(True, False),
    stock_models=['Heston'])
#   -- tables with basis functions
table_spots_Dim_BS_MaxCallr0_bf = _SmallDimensionTable(
    spots=[80, 100,  120], drift=(0.0,), use_payoff_as_input=(True, False))
table_Dim_Heston_MaxCallr0_bf = _SmallDimensionTable(
    stock_models=['Heston'], drift=(0.0,), use_payoff_as_input=(True, False))
#   -- true price
table_spots_Dim_MaxCallr0_ref = _DimensionTable(
    spots=[80, 100, 120], drift=(0.0,), algos=["EOP"],
    stock_models=['BlackScholes', 'Heston'])
#   -- tables to generate output tables
algos = ['NLSM', 'RFQI', 'RLSM', 'LSM', 'FQI', 'DOS', 'EOP', 'B']
table_spots_Dim_BS_MaxCallr0_gt1 = _DimensionTable(
    spots=[80, 100,  120], algos=algos, ridge_coeff=[1, np.nan, None],
    drift=(0.0,), use_payoff_as_input=(True, False,))
table_Dim_Heston_MaxCallr0_gt1 = _DimensionTable(
    stock_models=['Heston'], algos=algos, ridge_coeff=[1, np.nan, None],
    drift=(0.0,), use_payoff_as_input=(True, False))

# Heston with Var
table_Dim_HestonV_MaxCallr0 = _DimensionTable(
    algos=['NLSM', 'RFQI', 'RLSM', 'DOS'],
    stock_models=['HestonWithVar'], drift=(0.0,), use_payoff_as_input=(True, False))
#   -- tables with basis functions
table_Dim_HestonV_MaxCallr0_bf = _SmallDimensionTable(
    stock_models=['HestonWithVar'], drift=(0.0,), nb_stocks=(5, 10, 50,),
    use_payoff_as_input=(True, False))
#   -- true price
table_spots_Dim_HestonV_MaxCallr0_ref = _DimensionTable(
    drift=(0.0,), algos=["EOP"],
    stock_models=['HestonWithVar'])
#   -- tables to generate output tables
table_Dim_HestonV_MaxCallr0_gt1 = _DimensionTable(
    stock_models=['HestonWithVar'], algos=algos, ridge_coeff=[1, np.nan, None],
    drift=(0.0,), use_payoff_as_input=(True, False))


# RoughHeston model
table_Dim_RoughHeston_MaxCallr0 = _DimensionTable(
    nb_stocks=[5, 10, 50, 100], train_ITM_only=[False,],
    stock_models=['RoughHeston'], dividends=[0.1], drift=(0.05,), hurst=(0.05,),
    use_payoff_as_input=(False, True))
table_Dim_RoughHeston_MaxCallr0_do = _DimensionTable(
    algos=['DOS',], dividends=[0.1], drift=(0.05,), hurst=(0.05,),
    nb_stocks=[5, 10, 50, 100],
    use_payoff_as_input=(False, True), train_ITM_only=[False,],
    stock_models=['RoughHeston'])
table_Dim_RoughHeston_MaxCallr0_dopath = _DimensionTable(
    algos=['pathDOS',], dividends=[0.1], drift=(0.05,), hurst=(0.05,),
    nb_stocks=[5, 10, 50, 100], use_path=[True],
    use_payoff_as_input=(False, True), train_ITM_only=[False,],
    stock_models=['RoughHeston'])
table_Dim_RoughHeston_MaxCallr0_bf = _SmallDimensionTable(
    stock_models=['RoughHeston'], dividends=[0.1], drift=(0.05,), hurst=(0.05,),
    use_payoff_as_input=(False, True), nb_stocks=[5, 10, 50, 100],
    train_ITM_only=[False,],)
table_Dim_RoughHeston_MaxCallr0_RRLSM = _DimensionTable(
    algos=['RRLSM', 'RRFQI'], train_ITM_only=[False], nb_stocks=[5, 10, 50, 100],
    factors=[[0.0008, 0.11], [0.0001, 0.3]], hurst=(0.05,),
    stock_models=['RoughHeston'], dividends=[0.1], drift=(0.05,),
    use_payoff_as_input=(False, True))
table_Dim_RoughHeston_MaxCallr0_gt1 = _DimensionTable(
    train_ITM_only=[False,], nb_stocks=[5, 10, 50, 100],
    factors=[str((1., 1., 1.)), str([0.0001, 0.3])],
    stock_models=['RoughHeston'], hurst=(0.05,),
    algos=['NLSM', 'RFQI', 'RLSM', 'LSM', 'FQI', 'DOS', 'RRLSM', 'pathDOS'],
    ridge_coeff=[1, np.nan, None],
    dividends=[0.1], drift=(0.05,), use_payoff_as_input=(False, True))

# RoughHestonWithVar model
table_Dim_RoughHestonV_MaxCall = _DimensionTable(
    algos=['NLSM', 'RFQI', 'RLSM', 'DOS'],
    nb_stocks=[5, 10, 50, 100], train_ITM_only=[False,],
    stock_models=['RoughHestonWithVar'], dividends=[0.1], drift=(0.05,),
    hurst=(0.05,), use_payoff_as_input=(False, True))
table_Dim_RoughHestonV_MaxCall_dopath = _DimensionTable(
    algos=['pathDOS',], dividends=[0.1], drift=(0.05,), hurst=(0.05,),
    nb_stocks=[5, 10, 50, 100], use_path=[True],
    use_payoff_as_input=(False, True), train_ITM_only=[False,],
    stock_models=['RoughHestonWithVar'])
table_Dim_RoughHestonV_MaxCall_bf = _SmallDimensionTable(
    stock_models=['RoughHestonWithVar'], dividends=[0.1], drift=(0.05,),
    hurst=(0.05,), use_payoff_as_input=(False, True), nb_stocks=[5, 10, 50,],
    train_ITM_only=[False,],)
table_Dim_RoughHestonV_MaxCall_RRLSM = _DimensionTable(
    algos=['RRLSM', 'RRFQI'], train_ITM_only=[False], nb_stocks=[5, 10, 50, 100],
    factors=[[0.0008, 0.11],], hurst=(0.05,),
    stock_models=['RoughHestonWithVar'], dividends=[0.1], drift=(0.05,),
    use_payoff_as_input=(False, True))
table_Dim_RoughHestonV_MaxCall_gt1 = _DimensionTable(
    train_ITM_only=[False,], nb_stocks=[5, 10, 50, 100],
    stock_models=['RoughHestonWithVar'], hurst=(0.05,),
    algos=['NLSM', 'RFQI', 'RLSM', 'LSM', 'FQI', 'DOS', 'RRLSM', 'pathDOS'],
    ridge_coeff=[1, np.nan, None], factors=["(1.0, 1.0, 1.0)", "[0.0008, 0.11]"],
    dividends=[0.1], drift=(0.05,), use_payoff_as_input=(False, True))


# GeoPut
table_smallDim_BS_GeoPut = _VerySmallDimensionTable(
    payoffs=['GeometricPut'],
    nb_stocks=[100],
    algos=['RFQI',  ],
    stock_models=['Heston', ],
    use_payoff_as_input=(True, False))
table_smallDim_BS_GeoPut_BS = _VerySmallDimensionTable(
    payoffs=['GeometricPut'],
    nb_stocks=[5,10,20,50,100,],
    algos=['LSM', 'FQI', ],
    stock_models=['Heston', 'BlackScholes'],
    use_payoff_as_input=(True, False))
#   -- true price GeoPut
#      ATTENTION: after running, the payoff, nb_dates, nb_stocks, dividends,
#      volatility, need to
#      be changed in the metric file to get the correct tables
table_smallDim_BS_GeoPut_ref1 = _VerySmallDimensionTable(
    payoffs=['Put1Dim'], algos=["B"], nb_runs=1,
    nb_stocks=[1], nb_dates=[10000],
    volatilities=[0.2],
    stock_models=['BlackScholes'],)
table_smallDim_BS_GeoPut_ref2 = _VerySmallDimensionTable(
    payoffs=['Put1Dim'], algos=["B"], nb_runs=1,
    nb_stocks=[5], nb_dates=[10000],
    volatilities=[0.2],
    stock_models=['BlackScholes'],)
table_smallDim_BS_GeoPut_ref3 = _VerySmallDimensionTable(
    payoffs=['Put1Dim'], algos=["B"], nb_runs=1,
    nb_stocks=[10], nb_dates=[10000],
    volatilities=[0.2],
    stock_models=['BlackScholes'],)
table_smallDim_BS_GeoPut_ref4 = _VerySmallDimensionTable(
    payoffs=['Put1Dim'], algos=["B"], nb_runs=1,
    nb_stocks=[20], nb_dates=[10000],
    volatilities=[0.2],
    stock_models=['BlackScholes'],)
table_smallDim_BS_GeoPut_ref5 = _VerySmallDimensionTable(
    payoffs=['Put1Dim'], algos=["B"], nb_runs=1,
    nb_stocks=[50], nb_dates=[10000],
    volatilities=[0.2],
    stock_models=['BlackScholes'],)
table_smallDim_BS_GeoPut_ref6 = _VerySmallDimensionTable(
    payoffs=['Put1Dim'], algos=["B"], nb_runs=1,
    nb_stocks=[100], nb_dates=[10000],
    volatilities=[0.2],
    stock_models=['BlackScholes'],)
table_smallDim_BS_GeoPut_ref7 = _VerySmallDimensionTable(
    payoffs=['Put1Dim'], algos=["B"], nb_runs=1,
    nb_stocks=[500], nb_dates=[10000],
    volatilities=[0.2],
    stock_models=['BlackScholes'],)
table_smallDim_BS_GeoPut_ref8 = _VerySmallDimensionTable(
    payoffs=['Put1Dim'], algos=["B"], nb_runs=1,
    nb_stocks=[1000], nb_dates=[10000],
    volatilities=[0.2],
    stock_models=['BlackScholes'],)
table_smallDim_BS_GeoPut_ref9 = _VerySmallDimensionTable(
    payoffs=['Put1Dim'], algos=["B"], nb_runs=1,
    nb_stocks=[2000], nb_dates=[10000],
    volatilities=[0.2],
    stock_models=['BlackScholes'],)
# -- overview table
table_GeoPut_payoffs_gt1 = _DimensionTable(
    payoffs=['GeometricPut'], algos=algos,
    stock_models=['BlackScholes', 'Heston', ],
    nb_stocks=[5, 10, 20, 50, 100,],
    nb_dates=[10, 10000],
    dividends=[0.0,],
    volatilities=[0.2,],
    drift=(0.02,), use_payoff_as_input=(True, False))


# BasketCall BS
table_Dim_BS_BasktCallr0 = _DimensionTable(
    payoffs=['BasketCall'], algos=('NLSM', 'RFQI', 'RLSM', 'DOS'),
    drift=(0.0,), use_payoff_as_input=(True, False))
table_Dim_BS_BasktCallr0_bf = _SmallDimensionTable(
    payoffs=['BasketCall'], drift=(0.0,), use_payoff_as_input=(True, False))
#  -- true price
table_spots_Dim_BasktCallr0_ref = _DimensionTable(
    drift=(0.0,), algos=["EOP"], payoffs=['BasketCall'])
#  -- overview tables
table_BasketCall_payoffsr0_gt = _DimensionTable(
    payoffs=['BasketCall',], algos=algos,
    stock_models=['BlackScholes',],
    nb_stocks=[5, 10, 20, 50, 100, 500, 1000, 2000],
    drift=(0.0,), use_payoff_as_input=(False,))
table_BasketCall_payoffsr0_gt1 = _DimensionTable(
    payoffs=['BasketCall',], algos=algos,
    stock_models=['BlackScholes',],
    nb_stocks=[5, 10, 20, 50, 100, 500, 1000, 2000],
    drift=(0.0,), use_payoff_as_input=(True, False))


# other payoffs HestonWithVar
table_smallDim_HestonV_GeoPut = _VerySmallDimensionTable(
    payoffs=['GeometricPut'],
    algos=['NLSM', 'RFQI', 'RLSM', 'LSM', 'FQI', 'DOS', ],
    stock_models=['HestonWithVar',],
    use_payoff_as_input=(True, False))
table_GeoPut_HestonV_payoffs_gt = _DimensionTable(
    payoffs=['GeometricPut'], algos=algos,
    stock_models=['HestonWithVar', ],
    nb_stocks=[5, 10, 20,],
    dividends=[0.0,],
    volatilities=[0.2,],
    nb_dates=[10],
    drift=(0.02,), use_payoff_as_input=(False,))
table_GeoPut_HestonV_payoffs_gt1 = _DimensionTable(
    payoffs=['GeometricPut'], algos=algos,
    stock_models=['HestonWithVar', ],
    nb_stocks=[5, 10, 20,],
    dividends=[0.0,],
    volatilities=[0.2,],
    nb_dates=[10],
    drift=(0.02,), use_payoff_as_input=(False,True))


# many dates tables
algos = ['NLSM', 'RFQI', 'RLSM', 'LSM', 'DOS']
table_manyDates_BS_MaxCallr0_1 = _VerySmallDimensionTable(
    algos=algos, nb_stocks=[10, 50, ], nb_dates=[50, 100],
    drift=(0.0,), use_payoff_as_input=(True, False))
table_manyDates_BS_MaxCallr0_FQI = _VerySmallDimensionTable(
    algos=["FQI"], nb_stocks=[10, 50, ], nb_dates=[50, 100],
    drift=(0.0,), use_payoff_as_input=(True, False))
algos_ = ['NLSM', 'RFQI', 'RLSM', 'DOS']
table_manyDates_BS_MaxCallr0_2 = _VerySmallDimensionTable(
    algos=algos_, nb_stocks=[100, 500,], nb_dates=[50, 100],
    drift=(0.0,), use_payoff_as_input=(True, False))
table_manyDates_BS_MaxCallr0_ref = _VerySmallDimensionTable(
    algos= ['EOP'], nb_stocks=[10, 50, 100, 500,], nb_dates=[50, 100],
    drift=(0.0,), use_payoff_as_input=(False,))
table_manyDates_BS_MaxCallr0_gt = _VerySmallDimensionTable(
    algos=['NLSM', 'RFQI', 'RLSM', 'LSM', 'DOS', 'FQI', 'EOP'],
    nb_stocks=[10, 50, 100, 500,], nb_dates=[10, 50, 100],
    hidden_size=[20], ridge_coeff=[1, np.nan, None],
    drift=(0.0,), use_payoff_as_input=(True,))
table_manyDates_BS_MaxCallr0_gt1 = _VerySmallDimensionTable(
    algos=['NLSM', 'RFQI', 'RLSM', 'LSM', 'DOS', 'FQI', 'EOP'],
    nb_stocks=[10, 50, 100, 500,], nb_dates=[10, 50, 100],
    hidden_size=[20], ridge_coeff=[1, np.nan, None],
    drift=(0.0,), use_payoff_as_input=(True, False))
# -- many dates tables, BS with dividend
table_manyDates_BS_MaxCall_div_1 = _VerySmallDimensionTable(
    algos = ['NLSM', 'RLSM', 'LSM', 'DOS'],
    nb_stocks=[10, 50, ], nb_dates=[50, 100],
    dividends=[0.1], drift=(0.05,), use_payoff_as_input=(True,))
table_manyDates_BS_MaxCall_div_FQI = _VerySmallDimensionTable(
    algos=["FQI", "RFQI"], nb_stocks=[10, 50, ], nb_dates=[50, 100],
    dividends=[0.1], drift=(0.05,), use_payoff_as_input=(False,))
table_manyDates_BS_MaxCall_div_2 = _VerySmallDimensionTable(
    algos=['NLSM', 'RLSM', 'DOS'],
    nb_stocks=[100, 500,], nb_dates=[50, 100],
    dividends=[0.1], drift=(0.05,), use_payoff_as_input=(True, ))
table_manyDates_BS_MaxCall_div_FQI_2 = _VerySmallDimensionTable(
    algos=["RFQI"], nb_stocks=[100, 500, ], nb_dates=[50, 100],
    dividends=[0.1], drift=(0.05,), use_payoff_as_input=(False,))
table_manyDates_BS_MaxCall_div_gt1 = _VerySmallDimensionTable(
    algos=['NLSM', 'RFQI', 'RLSM', 'LSM', 'DOS', 'FQI',],
    nb_stocks=[10, 50, 100, 500,], nb_dates=[10, 50, 100],
    hidden_size=[20], ridge_coeff=[1, np.nan, None],
    dividends=[0.1], drift=(0.05,),
    use_payoff_as_input=(True, False))



# BS MinPut
table_spots_Dim_BS_MinPut = _DimensionTable(
    payoffs=["MinPut"],
    spots=[80, 100, 120], drift=(0.02,), use_payoff_as_input=(True, False))
table_spots_Dim_BS_MinPut_do = _DimensionTable(
    payoffs=["MinPut"],
    algos=['DOS',], drift=(0.02,), use_payoff_as_input=(True, False),
    spots=[80, 100, 120])
#   -- tables with basis functions
table_spots_Dim_BS_MinPut_bf = _SmallDimensionTable(
    payoffs=["MinPut"],
    spots=[80, 100, 120], drift=(0.02,), use_payoff_as_input=(True, False))
#   -- tables to generate output tables
algos = ['NLSM', 'RFQI', 'RLSM', 'LSM', 'FQI', 'DOS']
table_spots_Dim_BS_MinPut_gt = _DimensionTable(
    payoffs=["MinPut"],
    spots=[80, 100, 120], algos=algos, ridge_coeff=[1, np.nan, None],
    drift=(0.02,), use_payoff_as_input=(False,))
table_spots_Dim_BS_MinPut_gt1 = _DimensionTable(
    payoffs=["MinPut"],
    spots=[80, 100, 120], algos=algos, ridge_coeff=[1, np.nan, None],
    drift=(0.02,), use_payoff_as_input=(True, False,))


# BS MaxCall with dividend
table_Dim_BS_MaxCall_div = _DimensionTable(
    payoffs=['MaxCall'], algos=('NLSM', 'RFQI', 'RLSM', 'DOS'),
    dividends=[0.1],
    drift=(0.05,), use_payoff_as_input=(True, False))
table_Dim_BS_MaxCall_div_bf = _SmallDimensionTable(
    dividends=[0.1],
    payoffs=['MaxCall'], drift=(0.05,), use_payoff_as_input=(True, False))
table_Dim_BS_MaxCall_div_gt = _DimensionTable(
    payoffs=['MaxCall',], algos=algos, dividends=[0.1],
    drift=(0.05,), use_payoff_as_input=(False,))
table_Dim_BS_MaxCall_div_gt1 = _DimensionTable(
    payoffs=['MaxCall',], algos=algos, dividends=[0.1],
    drift=(0.05,), use_payoff_as_input=(True, False))


# Heston MinPut
table_spots_Dim_Heston_MinPut = _DimensionTable(
    payoffs=["MinPut"], stock_models=['Heston',],
    algos=('NLSM', 'RFQI', 'RLSM', 'DOS'),
    spots=[100], drift=(0.02,), use_payoff_as_input=(True, False))
#   -- tables with basis functions
table_spots_Dim_Heston_MinPut_bf = _SmallDimensionTable(
    payoffs=["MinPut"], stock_models=['Heston',],
    spots=[100], drift=(0.02,), use_payoff_as_input=(True, False))
#   -- tables to generate output tables
algos = ['NLSM', 'RFQI', 'RLSM', 'LSM', 'FQI', 'DOS']
table_spots_Dim_Heston_MinPut_gt = _DimensionTable(
    payoffs=["MinPut"], stock_models=['Heston',],
    spots=[100], algos=algos, ridge_coeff=[1, np.nan, None],
    drift=(0.02,), use_payoff_as_input=(False,))
table_spots_Dim_Heston_MinPut_gt1 = _DimensionTable(
    payoffs=["MinPut"], stock_models=['Heston',],
    spots=[100], algos=algos, ridge_coeff=[1, np.nan, None],
    drift=(0.02,), use_payoff_as_input=(True, False,))


# Heston MaxCall with dividend
table_Dim_Heston_MaxCall_div = _DimensionTable(
    payoffs=['MaxCall'], algos=('RFQI', ),
    nb_stocks=(1000,),
    dividends=[0.1], stock_models=['Heston',],
    drift=(0.05,), use_payoff_as_input=(True, False))
table_Dim_Heston_MaxCall_div_bf = _SmallDimensionTable(
    dividends=[0.1], stock_models=['Heston',],
    payoffs=['MaxCall'], drift=(0.05,), use_payoff_as_input=(True, False))
table_Dim_Heston_MaxCall_div_gt1 = _DimensionTable(
    stock_models=['Heston',],
    payoffs=['MaxCall',], algos=algos, dividends=[0.1],
    drift=(0.05,), use_payoff_as_input=(True, False))


# HestonV MinPut
table_spots_Dim_HestonV_MinPut = _DimensionTable(
    payoffs=["MinPut"], stock_models=['HestonWithVar',],
    algos=('NLSM', 'RFQI', 'RLSM', 'DOS'),
    spots=[100], drift=(0.02,), use_payoff_as_input=(True, False))
#   -- tables with basis functions
table_spots_Dim_HestonV_MinPut_bf = _SmallDimensionTable(
    payoffs=["MinPut"], stock_models=['HestonWithVar',], nb_stocks=(5, 10, 50,),
    spots=[100], drift=(0.02,), use_payoff_as_input=(True, False))
#   -- tables to generate output tables
table_spots_Dim_HestonV_MinPut_gt1 = _DimensionTable(
    payoffs=["MinPut"], stock_models=['HestonWithVar',],
    spots=[100], algos=algos, ridge_coeff=[1, np.nan, None],
    drift=(0.02,), use_payoff_as_input=(True, False,))


# HestonV MaxCall with dividend
table_Dim_HestonV_MaxCall_div = _DimensionTable(
    payoffs=['MaxCall'], algos=('NLSM', 'RFQI', 'RLSM', 'DOS'),
    dividends=[0.1], stock_models=['HestonWithVar',],
    drift=(0.05,), use_payoff_as_input=(True, False))
table_Dim_HestonV_MaxCall_div_bf = _SmallDimensionTable(
    dividends=[0.1], stock_models=['HestonWithVar',], nb_stocks=(5, 10, 50,),
    payoffs=['MaxCall'], drift=(0.05,), use_payoff_as_input=(True, False))
table_Dim_HestonV_MaxCall_div_gt1 = _DimensionTable(
    stock_models=['HestonWithVar',],
    payoffs=['MaxCall',], algos=algos, dividends=[0.1],
    drift=(0.05,), use_payoff_as_input=(True, False))




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
Tests for Sensitivity to Randomness of hidden layers 
'''
# ------------- Randomness Comparison (fixed paths)
SensRand_greeks_table1 = _DimensionTable(
    nb_runs=10, nb_epochs=[10],
    payoffs=["MaxCall"], volatilities=[0.2], drift=[0.02],
    strikes=[100], spots=[100], nb_dates=[10],
    hidden_size=[20], use_payoff_as_input=(False,), train_ITM_only=[True],
    algos=["RLSMSoftplus", "RLSMSoftplusReinit"], nb_stocks=[1],
    nb_paths=[100000])
SensRand_greeks_table1_1 = _DimensionTable(
    nb_runs=10, nb_epochs=[10, 30, 50],
    payoffs=["MaxCall"], volatilities=[0.2], drift=[0.02],
    strikes=[100], spots=[100], nb_dates=[10],
    hidden_size=[20], use_payoff_as_input=(True,), train_ITM_only=[True],
    algos=["NLSM"], nb_stocks=[1], nb_paths=[100000])



'''
Test for the FBM case of DOS
'''
# RNNLeastSquares
hurst = list(np.linspace(0, 1, 21))
hurst[0] = 0.01

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
        'FQI'
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
    nb_stocks=[5, 10, ],
    spots=[0], nb_epochs=[30],
    hurst=hurst1, train_ITM_only=[False],
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    use_payoff_as_input=[True],
    algos=[
        'DOS',
        'RLSM',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)
table_highdim_hurst_PD = _DefaultConfig(
    payoffs=['Max', 'Mean'],
    nb_stocks=[5, 10,],
    spots=[0], nb_epochs=[30],
    hurst=hurst1, train_ITM_only=[False],
    use_path=[True],
    use_payoff_as_input=[True],
    stock_models=['FractionalBrownianMotion'],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100],
    algos=[
        'pathDOS',
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
    use_payoff_as_input=[True],
    algos=[
        'RRLSM',
    ], nb_runs=10,
    representations=['TablePriceDuration']
)

factors1 = [str((1.,1.,1.))] + [str(x) for x in factors+factors0]
table_highdim_hurst_gt = _DefaultConfig(
    payoffs=['Identity', 'Max', 'Mean'],
    nb_stocks=[1, 5, 10, ],
    spots=[0], nb_epochs=[30],
    hurst=[0.05,], train_ITM_only=[False],
    stock_models=['FractionalBrownianMotion',],
    hidden_size=(20,), maturities=[1], nb_paths=[20000],
    nb_dates=[100], factors=factors1,
    use_payoff_as_input=[True, False],
    algos=[
        'DOS',
        'pathDOS',
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


'''
greeks computations
'''
algos = ["NLSM", "DOS", "RLSMSoftplus", "RFQI", "RFQISoftplus"]
test_table_greeks_1 = _DimensionTable(
    nb_runs=1, nb_epochs=[20],
    payoffs=["MinPut"], volatilities=[0.2], drift=[0.06],
    strikes=[36,], spots=[40], nb_dates=[10],
    hidden_size=[100], use_payoff_as_input=(True, False,), train_ITM_only=[True],
    algos=["RLSMSoftplus"], nb_stocks=[1], nb_paths=[100000])

algos = ["FQI", "RFQI"] +\
        ["RLSMSoftplus","RLSMElu", "RLSMSilu","RLSMGelu","RLSMTanh"]+\
        ["RLSM", "LSM", "NLSM", "DOS"]
table_greeks_1 = _DimensionTable(
    nb_runs=10, nb_epochs=[10],
    payoffs=["MinPut"], volatilities=[0.2], drift=[0.06],
    strikes=[36,40, 44], spots=[40], nb_dates=[10],
    hidden_size=[10], use_payoff_as_input=(True,), train_ITM_only=[True],
    algos=algos, nb_stocks=[1], nb_paths=[100000])
algos = ["NLSM", "DOS"]
table_greeks_1_2 = _DimensionTable(
    nb_runs=10, nb_epochs=[10],
    payoffs=["MinPut"], volatilities=[0.2], drift=[0.06],
    strikes=[36,40,44], spots=[40], nb_dates=[10],
    hidden_size=[10,], use_payoff_as_input=(True,), train_ITM_only=[True],
    algos=algos, nb_stocks=[1], nb_paths=[100000])

table_greeks_binomial = _DimensionTable(
    nb_runs=1, algos=["B"],
    payoffs=["Put1Dim"], volatilities=[0.2], drift=[0.06],
    strikes=[36,40,44], spots=[40], nb_dates=[10000, 50000],
    nb_stocks=[1])


spots = np.linspace(20, 60, 41)
table_greeks_plots = _DimensionTable(
    nb_runs=5, nb_epochs=[10],
    payoffs=["MinPut"],
    volatilities=[0.1, 0.2, 0.3, 0.4],
    maturities=[1, 0.5, 2, 4, 8],
    drift=[0.06],
    strikes=[40],
    spots=spots.tolist(), nb_dates=[10],
    hidden_size=[10], use_payoff_as_input=(True,), train_ITM_only=[True],
    algos=["RLSMSoftplus"], nb_stocks=[1], nb_paths=[100000])
table_greeks_plots_binomial = _DimensionTable(
    nb_runs=1,
    payoffs=["Put1Dim"],
    volatilities=[0.1, 0.2, 0.3, 0.4],
    maturities=[1, 0.5, 2, 4, 8],
    drift=[0.06],
    strikes=[40],
    spots=spots.tolist(), nb_dates=[10000],
    hidden_size=[10],
    algos=["B"], nb_stocks=[1],)


'''
upper bound computations
'''
table_price_lower_upper_1 = _DimensionTable(
    nb_runs=10, nb_epochs=[10],
    payoffs=["MaxCall"], volatilities=[0.2], drift=[0.05], dividends=[0.1],
    strikes=[100,], spots=[90, 100, 110], nb_dates=[9], maturities=[3],
    hidden_size=[100], use_payoff_as_input=(True,), train_ITM_only=[False],
    algos=["RLSMSoftplus",],
    nb_stocks=[2, 3, 5, 10, 20,],
    nb_paths=[100000])
table_price_lower_upper_1_1 = _DimensionTable(
    nb_runs=10, nb_epochs=[10],
    payoffs=["MaxCall"], volatilities=[0.2], drift=[0.05], dividends=[0.1],
    strikes=[100,], spots=[90, 100, 110], nb_dates=[9], maturities=[3],
    hidden_size=[-5], use_payoff_as_input=(True,), train_ITM_only=[False],
    algos=["RLSMSoftplus",], nb_stocks=[2, 3, 5, 10, 20, 30, 50, 100, 200, 500],
    nb_paths=[20000])

# ==============================================================================
test_table = _SmallDimensionTable(
    spots=[10], strikes=[10],
    algos=[
        'NLSM', 'LSM', 'DOS', 'FQI', 'RFQI', 'RLSM',
        "LSMLaguerre", "FQILaguerre", "LSMRidge", "RLSMRidge", "RLSMTanh",
        "FQIRidge", "RFQIRidge",
        "RRLSM", "RRLSMmix", "RFQITanh", "RRFQI",
        "LSPI",
        'FQIDeg1', 'LSMDeg1',
    ],
    nb_stocks=(5,), nb_dates=(5,), nb_paths=(100,),
    use_payoff_as_input=(True, False),
    nb_runs=1, factors=((0.001,0.001,0.001),),
    representations=['TablePriceDuration'],
)


test_table2 = _SmallDimensionTable(
    spots=[10], strikes=[10],
    algos=[
        'NLSM', 'LSM', 'DOS', 'FQI', 'RFQI', 'RLSM',
        "LSMLaguerre", "FQILaguerre", "LSMRidge", "RLSMRidge", "RLSMTanh",
        "RRLSM", "RRLSMmix", "RFQITanh", "RRFQI",
        "LSPI",
        'FQIDeg1', 'LSMDeg1',
        'pathDOS'
    ],
    stock_models=["Heston", "RoughHeston",
                  "HestonWithVar", "RoughHestonWithVar"],
    hurst=[0.25],
    nb_stocks=(5,), nb_dates=(5,), nb_paths=(100,),
    use_payoff_as_input=(True, False),
    nb_runs=1, factors=((0.001,0.001,0.001),),
    representations=['TablePriceDuration'],
)








