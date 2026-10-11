import torch
import pytest
@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)
