import importlib

def _get_absolute_mapping(name: str):
    # in this case, the `name` should be the fully qualified name of the class
    # e.g., `ocpmodels.tasks.base_task.BaseTask`
    # we can use importlib to get the module (e.g., `ocpmodels.tasks.base_task`)
    # and then import the class (e.g., `BaseTask`)

    module_name = ".".join(name.split(".")[:-1])
    class_name = name.split(".")[-1]

    try:
        module = importlib.import_module(module_name)
    except (ModuleNotFoundError, ValueError) as e:
        raise RuntimeError(
            f"Could not import module `{module_name}` for import `{name}`"
        ) from e

    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise RuntimeError(
            f"Could not import class `{class_name}` from module `{module_name}`"
        ) from e

    
class Registry:
    r"""Class for registry object which acts as central source of truth."""
    mapping = {
        "filter_name_mapping": {},
        "criterion_name_mapping": {},
        "optimizer_name_mapping": {},
        "task_name_mapping": {},
        "model_name_mapping": {},
        "rawdataloader_name_mapping": {},
        "state": {},
    }

    @classmethod
    def register_filter(cls, name):
        r"""Register a new data filter to registry with key 'name'
        Args:
            name: Key with which the filter will be registered.
        Usage::
            from common.registry import registry
            @registry.register_filter("lowpass_filter")
            class LowpassFilter:
        """

        def wrap(func):
            cls.mapping["filter_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_criterion(cls, name):
        r"""Register a new criterion to registry with key 'name'
        Args:
            name: Key with which the criterion will be registered.
        Usage::
            from common.registry import registry
            @registry.register_criterion("MSE")
            def mse_loss():
        """

        def wrap(func):
            cls.mapping["criterion_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_optimizer(cls, name):
        r"""Register a new optimizer to registry with key 'name'
        Args:
            name: Key with which the optimizer will be registered.
        Usage::
            from common.registry import registry
            @registry.register_optimizer("Adam")
            def adam_optim():
        """

        def wrap(func):
            cls.mapping["optimizer_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_task(cls, name):
        r"""Register a new task to registry with key 'name'
        Args:
            name: Key with which the task will be registered.
        Usage::
            from common.registry import registry
            @registry.register_task("train")
            class BaseTrainer:
        """

        def wrap(func):
            cls.mapping["task_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_model(cls, name):
        r"""Register a new model to registry with key 'name'
        Args:
            name: Key with which the model will be registered.
        Usage::
            from common.registry import registry
            @registry.register_model("LSTM4EMG")
            class LSTM4EMG(nn.Module):
        """

        def wrap(func):
            cls.mapping["model_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register_rawdataloader(cls, name):
        r"""Register a new raw data loader to registry with key 'name'
        Args:
            name: Key with which the raw data loader will be registered.
        Usage::
            from common.registry import registry
            @registry.register_rawdataloader("Ninapro")
            def ninapro_loader(config: dict):
        """

        def wrap(func):
            cls.mapping["rawdataloader_name_mapping"][name] = func
            return func

        return wrap

    @classmethod
    def register(cls, name, obj):
        r"""Register an item to registry with key 'name'

        Args:
            name: Key with which the item will be registered.

        Usage::

            from common.registry import registry

            registry.register("config", {})
        """
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    @classmethod
    def __import_error(cls, name: str, mapping_name: str):
        kind = mapping_name[: -len("_name_mapping")]
        mapping = cls.mapping.get(mapping_name, {})
        existing_keys = list(mapping.keys())

        existing_cls_path = (
            mapping.get(existing_keys[-1], None) if existing_keys else None
        )
        if existing_cls_path is not None:
            existing_cls_path = f"{existing_cls_path.__module__}.{existing_cls_path.__qualname__}"
        else:
            existing_cls_path = "data_preprocessing.filter.LowpassFilter"

        existing_keys = [f"'{name}'" for name in existing_keys]
        existing_keys = (
            ", ".join(existing_keys[:-1]) + " or " + existing_keys[-1]
        )
        existing_keys_str = (
            f" (one of {existing_keys})" if existing_keys else ""
        )
        return RuntimeError(
            f"Failed to find the {kind} '{name}'. "
            f"You may either use a {kind} from the registry{existing_keys_str} "
            f"or provide the full import path to the {kind} (e.g., '{existing_cls_path}')."
        )

    @classmethod
    def get_class(cls, name: str, mapping_name: str):
        existing_mapping = cls.mapping[mapping_name].get(name, None)
        if existing_mapping is not None:
            return existing_mapping

        # mapping be class path of type `{module_name}.{class_name}` (e.g., `data_preprocessing.filter.LowpassFilter`)
        if name.count(".") < 1:
            raise cls.__import_error(name, mapping_name)

        try:
            return _get_absolute_mapping(name)
        except RuntimeError as e:
            raise cls.__import_error(name, mapping_name) from e

    @classmethod
    def get_filter_class(cls, name):
        return cls.get_class(name, "filter_name_mapping")

    @classmethod
    def get_criterion(cls, name):
        return cls.get_class(name, "criterion_name_mapping")

    @classmethod
    def get_optimizer(cls, name):
        return cls.get_class(name, "optimizer_name_mapping")

    @classmethod
    def get_task_class(cls, name):
        return cls.get_class(name, "task_name_mapping")

    @classmethod
    def get_model_class(cls, name):
        return cls.get_class(name, "model_name_mapping")

    @classmethod
    def get_rawdataloader(cls, name):
        return cls.get_class(name, "rawdataloader_name_mapping")

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        r"""Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retreived.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for cgcnn's
                               internal operations. Default: False
        Usage::

            from common.registry import registry

            config = registry.get("config")
        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
            "writer" in cls.mapping["state"]
            and value == default
            and no_warning is False
        ):
            cls.mapping["state"]["writer"].write(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def unregister(cls, name):
        r"""Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)

registry = Registry()