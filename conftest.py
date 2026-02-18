# Block ROS 2 launch_testing plugins that try to load at import-time
# and fail because pyyaml isn't available in this venv.
#
# The plugins are registered via setuptools entry-points installed
# system-wide by the ROS 2 Humble packages. We must prevent the
# import from even being attempted by monkeypatching importlib.metadata
# *before* pluggy's ``load_setuptools_entrypoints`` iterates.
import importlib.metadata as _imd

_orig_entry_points = _imd.entry_points


def _filtered_entry_points(**kwargs):
    """Drop ROS 2 pytest11 entry points that would break this project."""
    result = _orig_entry_points(**kwargs)
    if kwargs.get("group") == "pytest11" or not kwargs:
        if hasattr(result, "select"):
            result = result.select(group="pytest11") if not kwargs else result
        blocked = {"launch_testing", "launch_testing_ros"}
        if isinstance(result, dict):
            result = {k: [e for e in v if e.name not in blocked] for k, v in result.items()}
        elif isinstance(result, list):
            result = [e for e in result if getattr(e, "name", None) not in blocked]
        else:
            try:
                result = type(result)(e for e in result if e.name not in blocked)
            except Exception:
                pass
    return result


_imd.entry_points = _filtered_entry_points


collect_ignore_glob = []


def pytest_configure(config):
    pm = config.pluginmanager
    for name in (
        "launch_testing",
        "launch_testing_ros",
        "launch_testing.pytest.ini",
        "launch_testing_ros_pytest_entrypoint",
    ):
        mod = pm.get_plugin(name)
        if mod is not None:
            pm.unregister(mod)
