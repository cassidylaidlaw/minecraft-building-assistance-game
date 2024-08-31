import pytest

try:
    from sacred.config.custom_containers import DogmaticDict, DogmaticList

    from mbag.rllib.sacred_utils import convert_dogmatics_to_standard
except ImportError:
    pass


@pytest.mark.uses_sacred
def test_dogmatic_dict_conversion():
    dogmatic_dict = DogmaticDict()
    dogmatic_dict["key"] = "value"
    converted = convert_dogmatics_to_standard(dogmatic_dict)
    assert isinstance(converted, dict)
    assert converted == {"key": "value"}


@pytest.mark.uses_sacred
def test_dogmatic_list_conversion():
    dogmatic_list = DogmaticList(["item1", "item2"])
    converted = convert_dogmatics_to_standard(dogmatic_list)
    assert isinstance(converted, list)
    assert converted == ["item1", "item2"]


@pytest.mark.uses_sacred
def test_standard_objects():
    data = {
        "list": [1, 2, 3],
        "dict": {"a": 1, "b": 2},
        "string": "hello",
        "int": 10,
    }
    converted = convert_dogmatics_to_standard(data)
    assert converted == data


@pytest.mark.uses_sacred
def test_nested_combinations():
    dogmatic_dict = DogmaticDict()
    dogmatic_dict["inner_key"] = "value"
    nested_structure = {
        "dogmatic_list": DogmaticList([dogmatic_dict]),
        "standard_list": [1, 2, 3],
        "string": "hello",
    }
    converted = convert_dogmatics_to_standard(nested_structure)
    expected_structure = {
        "dogmatic_list": [{"inner_key": "value"}],
        "standard_list": [1, 2, 3],
        "string": "hello",
    }
    assert converted == expected_structure
