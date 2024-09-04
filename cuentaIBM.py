from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="",
    set_as_default=True,
    # Use `overwrite=True` if you're updating your token.
    overwrite=True,
)
