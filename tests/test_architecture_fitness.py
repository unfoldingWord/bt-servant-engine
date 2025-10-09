"""Architectural fitness functions to enforce clean architecture principles.

These tests ensure the codebase maintains its hexagonal/onion architecture
by preventing common anti-patterns and structural violations.
"""

import re
from pathlib import Path


def test_no_python_modules_at_root():
    """Prevent re-export shims at root, but allow legitimate entry points.

    The key distinction:

    ❌ SHIM (Anti-pattern):
        # brain.py - OTHER CODE IMPORTS FROM THIS
        from package.module import function

        External code does: from brain import function
        Problem: Re-exports internals, creates backdoor around architecture
        Import direction: External → shim → internal (pulling internals OUT)

    ✅ ENTRY POINT (Legitimate):
        # main.py - NOTHING IMPORTS FROM THIS (leaf node)
        from package.module import function
        app.run()

        Nothing imports from main.py - it's a consumer, not a provider
        Import direction: main.py → internal (one-way, inward only)

    Allowed root-level files:
    - setup.py: Packaging/installation
    - conftest.py: Pytest configuration
    - main.py, __main__.py: Application entry points (must be leaf nodes)
    """
    root = Path(__file__).parent.parent
    python_files = list(root.glob("*.py"))

    # Legitimate files at root
    allowed = {"setup.py", "conftest.py", "main.py", "__main__.py"}
    violations = [f.name for f in python_files if f.name not in allowed]

    assert not violations, (
        f"Unexpected Python modules at root: {violations}\n\n"
        "Only entry points and config files allowed at root.\n"
        "Re-export shims must live inside bt_servant_engine/ package where "
        "import-linter can enforce architectural boundaries.\n\n"
        "See test docstring for distinction between shims and entry points."
    )


def test_root_files_are_not_imported():
    """Ensure root-level .py files are leaf nodes (entry points), not shims.

    This test catches the actual anti-pattern: a root-level file being
    imported FROM by other code (making it a shim/re-export facade).

    Entry points (main.py) should be consumers, not providers.
    """
    root = Path(__file__).parent.parent
    python_files = list(root.glob("*.py"))

    # Files that are legitimately imported (packaging, test config)
    allowed_imports = {"setup", "conftest"}

    # Check if any non-allowed root file is imported by package code
    violations = []
    for py_file in python_files:
        module_name = py_file.stem  # filename without .py

        if module_name in allowed_imports:
            continue

        # Search for imports of this module in the package
        package_dir = root / "bt_servant_engine"
        if package_dir.exists():
            for src_file in package_dir.rglob("*.py"):
                content = src_file.read_text()
                # Look for: "from module import" or "import module"
                import_patterns = [
                    rf"^from {module_name} import",
                    rf"^import {module_name}\b",
                ]
                for pattern in import_patterns:
                    if re.search(pattern, content, re.MULTILINE):
                        violations.append(
                            f"{src_file.relative_to(root)} imports from {py_file.name}"
                        )

    assert not violations, (
        "Root-level files are being imported (shim pattern detected):\n"
        + "\n".join(f"  - {v}" for v in violations)
        + "\n\n"
        "Entry points should be leaf nodes - nothing should import FROM them.\n"
        "If code needs to import something, it should live inside the package."
    )


def test_no_direct_chromadb_imports_in_services():
    """Services layer must not import chromadb directly.

    ChromaDB access should go through the ChromaPort adapter to maintain
    the dependency inversion principle.
    """
    services_dir = Path(__file__).parent.parent / "bt_servant_engine" / "services"

    violations = []
    for py_file in services_dir.rglob("*.py"):
        content = py_file.read_text()
        if "import chromadb" in content or "from chromadb" in content:
            violations.append(py_file.relative_to(services_dir))

    assert not violations, (
        f"Services layer imports chromadb directly: {violations}\n"
        "Use bt_servant_engine.core.ports.ChromaPort instead."
    )


def test_no_fastapi_in_core():
    """Core layer must not import FastAPI.

    The core is framework-agnostic and should not depend on web frameworks.
    """
    core_dir = Path(__file__).parent.parent / "bt_servant_engine" / "core"

    violations = []
    for py_file in core_dir.rglob("*.py"):
        content = py_file.read_text()
        if "import fastapi" in content or "from fastapi" in content:
            violations.append(py_file.relative_to(core_dir))

    assert not violations, (
        f"Core layer imports FastAPI: {violations}\n" "Core must remain framework-agnostic."
    )


def test_no_openai_in_core():
    """Core layer must not import OpenAI SDK.

    The core defines domain models and shouldn't depend on external SDKs.
    """
    core_dir = Path(__file__).parent.parent / "bt_servant_engine" / "core"

    violations = []
    for py_file in core_dir.rglob("*.py"):
        content = py_file.read_text()
        if "import openai" in content or "from openai" in content:
            violations.append(py_file.relative_to(core_dir))

    assert not violations, (
        f"Core layer imports OpenAI SDK: {violations}\n"
        "Core must not depend on external service SDKs."
    )
