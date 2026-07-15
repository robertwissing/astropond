"""PKDGRAV3 glue for the sph_interp library.

PKDGRAV3 reads and writes gasoline/tipsy format natively (achInFile reads the
tipsy IC; snapshots are written as tipsy `{name}.{step:05d}`), so there is NO IC
format conversion -- the suite's tipsy IC is fed directly. The only PKDGRAV3
artefact generated here is its `.par` parameterfile (see
`pkdgrav3_param.write_run_param`).
"""
