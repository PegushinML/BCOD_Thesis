from distutils.core import setup
import bayesian_online_changepoint_detection

setup(name='bayesian_online_changepoint_detection',
      version=bayesian_online_changepoint_detection.__version__,
      description='Bayesian online changepoint detection algorithm',
      author='Pegushin Maxim',
      url='https://github.com/PegushinML/BCOD_Thesis/script',
      packages = ['bayesian_online_changepoint_detection'],
      requires=['scipy', 'numpy']
     )
