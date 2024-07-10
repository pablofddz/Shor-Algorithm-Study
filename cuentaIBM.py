from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="5d799f890604e93d97df6f667a44d946c1edbe9c604e559b89711faf593eb0c1895286d9a32f0bd19a25d1c29e854504fedf27941d797cfc3d009df7e1e9471c",
    set_as_default=True,
    # Use `overwrite=True` if you're updating your token.
    overwrite=True,
)
