from abc import ABC, abstractmethod


class ImageMatcher(ABC):
    @abstractmethod
    def match(self, image0, image1):
        """Return matched points and method-specific confidence values."""
        raise NotImplementedError
