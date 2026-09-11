# NeuRelease

A fast neural parser for torrent and release names, with Python bindings over a C++23 library.

It reads a release name, returns every field it finds with the span each was read from and a
confidence, and classifies the work: film or series, music, game, software or book, and whether it
is anime.

```python
from neurelease import Parser

parser = Parser()
print(parser.parse("Ted.Lasso.S03E03.4-5-1.1080p.ATVP.WEB-DL.DDP5.1.H.264-NTb").to_dict())
# {'title': 'Ted Lasso', 'episode_title': '4-5-1', 'season': 3, 'episode': 3,
#  'screen_size': '1080p', 'source': 'WEB-DL', 'video_codec': 'H264', 'audio_codec': 'DDP',
#  'audio_channels': '5.1', 'streaming_service': 'ATVP', 'release_group': 'NTb',
#  'type': 'episode', 'content': 'series', 'medium': 'video', 'adult': False, 'anime': False,
#  'numbering': 'season_episode', 'confidence': {...}}
```

`Parser()` needs no paths: the wheel bundles the compiled library and the model, so it works from
any directory. `parse_batch` runs many names at once across several threads and returns them in
input order.

Inference is int8 on the CPU, with AVX2, AVX-VNNI and AVX-512 kernels selected at load time by
CPUID. A name parses in roughly 2,400 microseconds through this binding on a desktop Ryzen, about
three times faster than GuessIt, and considerably less per name in a batch.

**[Try it in the browser](https://neurelease-demo.vercel.app)** ·
[Source and documentation](https://github.com/bonejay/neurelease) ·
[What a result contains, field by field](https://github.com/bonejay/neurelease/blob/main/docs/RESULT.md)

MIT licensed. PCRE2 is statically linked into the bundled library; its notice travels in the wheel.
