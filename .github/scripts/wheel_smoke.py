"""Import the installed wheel and parse one name: run from outside the repository, so nothing can
fall back to the build tree. Every wheel the publish workflow builds passes through here, on the
platform it is for, before anything reaches PyPI."""

import neurelease

result = neurelease.Parser().parse("Ted.Lasso.S03E03.4-5-1.1080p.ATVP.WEB-DL.DDP5.1.H.264-NTb")
fields = result.to_dict()
assert fields["title"] == "Ted Lasso", fields
assert fields["season"] == 3 and fields["episode"] == 3, fields
assert fields["anime"] is False, fields
print("wheel works:", neurelease.__version__, neurelease.__file__)
print(fields)
