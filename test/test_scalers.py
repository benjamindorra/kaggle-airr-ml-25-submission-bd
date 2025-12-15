from ..src.scalers import get_scalers

def test_get_scalers():
    
    
    # Test get_scalers function
    assert get_scalers() == {"categorical": None, "numerical": None}
