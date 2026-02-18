# Block ROS 2 launch_testing plugins that are registered system-wide
# and conflict with this project's test suite.
collect_ignore_glob = []


def pytest_configure(config):
    pm = config.pluginmanager
    for name in ("launch_testing", "launch_testing_ros"):
        mod = pm.get_plugin(name)
        if mod is not None:
            pm.unregister(mod)
    # Also try entrypoint names
    for name in (
        "launch_testing.pytest.ini",
        "launch_testing_ros_pytest_entrypoint",
    ):
        mod = pm.get_plugin(name)
        if mod is not None:
            pm.unregister(mod)
