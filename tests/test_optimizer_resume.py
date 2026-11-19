import pytest

torch = pytest.importorskip("torch")


def _run(model, opt, batches):
    for x, y in batches:
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        opt.step()


def test_resumed_training_matches_one_run(tmp_path):
    from mvrag.agg_train import restore_optimizer, save_optimizer

    torch.manual_seed(0)
    batches = [(torch.randn(4, 3), torch.randn(4, 2)) for _ in range(6)]
    init = torch.nn.Linear(3, 2).state_dict()

    one = torch.nn.Linear(3, 2)
    one.load_state_dict(init)
    _run(one, torch.optim.AdamW(one.parameters(), lr=0.1), batches)

    first = torch.nn.Linear(3, 2)
    first.load_state_dict(init)
    opt = torch.optim.AdamW(first.parameters(), lr=0.1)
    _run(first, opt, batches[:3])
    save_optimizer(opt, tmp_path)

    second = torch.nn.Linear(3, 2)  # a fresh process: new model and optimizer
    second.load_state_dict(first.state_dict())
    opt2 = torch.optim.AdamW(second.parameters(), lr=0.1)
    assert restore_optimizer(opt2, tmp_path)
    _run(second, opt2, batches[3:])

    for a, b in zip(one.parameters(), second.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_restore_without_saved_state(tmp_path):
    from mvrag.agg_train import restore_optimizer

    model = torch.nn.Linear(3, 2)
    assert not restore_optimizer(torch.optim.AdamW(model.parameters()), tmp_path)
