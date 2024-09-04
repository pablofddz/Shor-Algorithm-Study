from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="5d156b588e1f2a8a4ce8b95174acd58fac04edafb1337dcb39ca849c69abf1507f7a69d094518a2e0ecb030bc56456f2d6c1015330394b4eee98ea9f820c7a58",
    set_as_default=True,
    # Use `overwrite=True` if you're updating your token.
    overwrite=True,
)
