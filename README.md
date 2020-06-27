# PicoTree

PicoTree is a small C++ header only library that provides several data structures that can be used for range searches and nearest neighbor searches. Created simply for fun, the first thing that was added was a [Range Tree](https://en.wikipedia.org/wiki/Range_tree). [nanoflann](https://github.com/jlblancoc/nanoflann) was used as a reference for very fast searches, but comparing to it was a bit awkward because it doesn't provide range query support. Obviously, this is reason enough to create your own [KdTree](https://en.wikipedia.org/wiki/K-d_tree) and a small library was born.
