from jina.executors import BaseExecutor


def test_clip():
    ex = BaseExecutor.load_config('../../config.yml')
    assert ex.default_batch_size == 32
    assert len(ex.default_traversal_paths) == 1
    assert ex.default_traversal_paths[0] == 'r'
    assert ex.device == 'cpu'
    assert ex.is_updated is False
