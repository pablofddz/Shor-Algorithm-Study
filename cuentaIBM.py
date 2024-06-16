from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="bc262bd6781ce7bde62961afe273a3bf528ad3aedde77cf6af5dd2862036dff0a0872815a2abc8f2c86459d804b6e44922a36a14b25fdc4cb86d8e435e93314c",
    set_as_default=True,
    # Use `overwrite=True` if you're updating your token.
    overwrite=True,
)