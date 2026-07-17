"""Holds all laboratory related exceptions."""


class LabException(Exception):
    """Base exception for all laboratory related exceptions."""


class LabNotImplementedException(LabException):
    """Exception describing a missing implementation."""

class LabCommandLineException(LabException):
    """Exception describing a invalid command line parameter."""

class LabConfigException(LabException):
    """Exception describing an invalid configuration."""


class LabJsonException(LabException):
    """Exception describing a malformed or unexpected JSON object."""


class LabConvertException(LabException):
    """Exception describing an abstract conversion error."""


class LabDataException(LabException):
    """Exception describing unexpected data."""


class LabParameterException(LabException):
    """Exception describing unexpected parameter."""
