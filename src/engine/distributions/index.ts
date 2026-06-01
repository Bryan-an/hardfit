import type { Distribution } from '../types'
import { exponential } from './exponential'
import { gamma } from './gamma'
import { lognormal } from './lognormal'
import { normal } from './normal'
import { weibull } from './weibull'

export const DISTRIBUTIONS: readonly Distribution[] = [
  normal,
  lognormal,
  exponential,
  gamma,
  weibull,
]
