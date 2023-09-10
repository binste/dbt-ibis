# How to cut a new release
1. Make certain your branch is in sync with head and then create a new release branch:

        git pull origin main
        git switch -c version_0.2.0

2. Update version to, e.g. 0.2.0 in `dbt_ibis/__init__.py`

3. Commit change and push:

        git add . -u
        git commit -m "MAINT: Bump version to 0.2.0"
        git push

4. Run test suite again after commit above to make sure everything passes:

        hatch run linters
        hatch run tests

5. Build source & wheel distributions:

        hatch clean  # clean old builds & distributions
        hatch build  # create a source distribution and universal wheel

6. publish to PyPI (Requires correct PyPI owner permissions):

        hatch publish

7. Merge release branch into main

8. On main, tag the release:

        git tag -a v0.2.0 -m "Version 0.2.0 release"
        git push v0.2.0

9. Add release in https://github.com/binste/dbt-ibis/releases and select the version tag

10. Update version to e.g. 0.3.0dev in `dbt_ibis/__init__.py` in new branch

        git switch -c maint_0.3.0dev

11. Commit change and push:

        git add . -u
        git commit -m "MAINT: Bump version to 0.3.0dev"
        git push

12. Merge maintenance branch into main
