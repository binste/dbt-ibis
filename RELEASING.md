# How to cut a new release
1. Make certain your branch is in sync with head and then create a new release branch:

        git pull origin main
        git switch -c version_0.2.0

2. Update version to, e.g. 0.2.0 in `dbt_ibis/__init__.py`

3. Commit change and push:

        git add . -u
        git commit -m "MAINT: Bump version to 0.2.0"
        git push

4. Merge release branch into main, make sure that all required checks pass

5. On main, build source & wheel distributions:

        git checkout main
        git pull
        hatch clean  # clean old builds & distributions
        hatch build  # create a source distribution and universal wheel

6. Publish to PyPI (Requires correct PyPI owner permissions):

        hatch publish

7. On main, tag the release:

        git tag -a v0.2.0 -m "Version 0.2.0 release"
        git push origin v0.2.0

8. Add release in https://github.com/binste/dbt-ibis/releases and select the version tag

9. Update version to e.g. 0.3.0dev in `dbt_ibis/__init__.py` in new branch

        git switch -c maint_0.3.0dev

10. Commit change and push:

        git add . -u
        git commit -m "MAINT: Bump version to 0.3.0dev"
        git push

11. Merge maintenance branch into main
