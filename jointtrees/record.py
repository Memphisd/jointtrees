"""Record: a dict whose string keys are also attributes.  Standard library only."""


class Record(dict):
    """A dict whose string keys are also attributes: P.filter[3].u is P['filter'][3]['u']."""
    __slots__ = ()

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            there = ", ".join(map(str, self)) or "none"
            raise AttributeError(f'no parameter {name!r} here, there are: {there}') from None

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None
