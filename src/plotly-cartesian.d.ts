// No `@types/plotly.js-cartesian-dist-min` exists, so we alias the cartesian bundle's
// types to `@types/plotly.js` (the bundle is a subset of the same runtime API). The
// `export =` form matches the CommonJS shape of the module and works under the repo's
// `moduleResolution: bundler` + `esModuleInterop`, the same as the @stdlib packages.
declare module 'plotly.js-cartesian-dist-min' {
  import * as Plotly from 'plotly.js'
  export = Plotly
}
