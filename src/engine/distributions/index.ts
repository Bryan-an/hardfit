import type { Distribution } from '../types'
import { beta } from './beta'
import { betaprime } from './betaprime'
import { cauchy } from './cauchy'
import { chi } from './chi'
import { chisquare } from './chisquare'
import { cosine } from './cosine'
import { discreteUniform } from './discrete-uniform'
import { exponential } from './exponential'
import { fisherF } from './fisher-f'
import { frechet } from './frechet'
import { gamma } from './gamma'
import { geometric } from './geometric'
import { gumbel } from './gumbel'
import { invgamma } from './invgamma'
import { laplace } from './laplace'
import { levy } from './levy'
import { logistic } from './logistic'
import { lognormal } from './lognormal'
import { negativeBinomial } from './negative-binomial'
import { normal } from './normal'
import { pareto } from './pareto'
import { poisson } from './poisson'
import { rayleigh } from './rayleigh'
import { studentT } from './student-t'
import { uniform } from './uniform'
import { weibull } from './weibull'

export const DISTRIBUTIONS: readonly Distribution[] = [
  normal,
  lognormal,
  exponential,
  gamma,
  weibull,
  // M2.3 Batch A — drop-in continuous distributions.
  uniform,
  rayleigh,
  pareto,
  laplace,
  logistic,
  gumbel,
  cauchy,
  frechet,
  // M2.3 Batch B — drop-in continuous distributions.
  levy,
  chisquare,
  chi,
  invgamma,
  betaprime,
  cosine,
  beta,
  // M2.3 Batch C — discrete distributions.
  poisson,
  geometric,
  negativeBinomial,
  discreteUniform,
  // M2.3 Batch D — multi-parameter MLE via the vendored Nelder–Mead optimizer.
  studentT,
  fisherF,
]
