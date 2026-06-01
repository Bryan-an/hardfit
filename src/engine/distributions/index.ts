import type { Distribution } from '../types'
import { cauchy } from './cauchy'
import { exponential } from './exponential'
import { frechet } from './frechet'
import { gamma } from './gamma'
import { gumbel } from './gumbel'
import { laplace } from './laplace'
import { logistic } from './logistic'
import { lognormal } from './lognormal'
import { normal } from './normal'
import { pareto } from './pareto'
import { rayleigh } from './rayleigh'
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
]
