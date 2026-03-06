###################################################################################
#   Copyright (c) 2021 STMicroelectronics.
#   All rights reserved.
#   This software is licensed under terms that can be found in the LICENSE file in
#   the root directory of this software component.
#   If no LICENSE file comes with this software, it is provided AS-IS.
###################################################################################
"""
STM AI runner - Utilities
"""

import sys
import logging
from io import StringIO
from typing import Union, List


from colorama import init, Fore, Style


_LOGGER_NAME_ = 'STMAI-RUNNER'
STMAI_RUNNER_LOGGER_NAME = _LOGGER_NAME_


class ColorFormatter(logging.Formatter):
    """Color Formatter"""  # noqa: DAR101,DAR201,DAR401

    COLORS = {
        "WARNING": (Fore.YELLOW + Style.BRIGHT, 'W'),
        "ERROR": (Fore.RED + Style.BRIGHT, 'E'),
        "DEBUG": (Fore.CYAN, 'D'),
        "INFO": (Fore.GREEN, 'I'),
        "CRITICAL": (Fore.RED + Style.BRIGHT, 'C')
    }

    def __init__(self, with_prefix=False, color=True, fmt=None, datefmt=None, style='%'):
        self.with_prefix = with_prefix
        self.color = color
        super(ColorFormatter, self).__init__(fmt, datefmt, style)

    def format(self, record):
        color, lname = self.COLORS.get(record.levelname, '')
        header = ':' + record.name if self.with_prefix else ''
        color = color if self.color else ''
        # record.name = ''
        record.levelname = color + '[' + lname + header + ']' + Style.RESET_ALL
        record.msg = color + str(record.msg) + Style.RESET_ALL
        return logging.Formatter.format(self, record)


def get_logger(name: str = _LOGGER_NAME_, level=logging.WARNING, color=True, with_prefix=False):
    """Utility function to create a logger object"""  # noqa: DAR101,DAR201,DAR401

    logger = logging.getLogger(name)

    if not logger.propagate and logger.hasHandlers():
        # logger 'name' already created
        return logger

    if color:
        init()

    logger.setLevel(level)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    color_formatter = ColorFormatter(fmt="%(levelname)s %(message)s", color=color, with_prefix=with_prefix)

    console.setFormatter(color_formatter)
    logger.addHandler(console)
    logger.propagate = False

    return logger


def set_log_level(level: Union[str, int] = logging.DEBUG, logger: Union[logging.Logger, None] = None):
    """Set the log level of the module"""  # noqa: DAR101,DAR201,DAR401

    if isinstance(level, str):
        level = level.upper()
    level = logging.getLevelName(level)

    if logger is None:
        logger = get_logger(_LOGGER_NAME_)
    logger.setLevel(level)
    if logger.handlers:
        logger.handlers[0].setLevel(level)


class TableWriter(StringIO):
    """Pretty-print tabular data (table form)"""

    N_SPACE = 2

    def __init__(self, indent: int = 0, csep: str = ' '):
        """Create the Table instance"""  # noqa: DAR101,DAR201
        self._header = []  # type: List[str]
        self._notes = []  # type: List[str]
        self._datas = []  # type: List[Union[List[str], str]]
        self._title = ''  # type: str
        self._fmt = ''  # type: str
        self._sizes = []  # type: List[int]
        self._indent = int(max(indent, 0))
        self._csep = csep
        super(TableWriter, self).__init__()

    def set_header(self, items: Union[List[str], str]):
        """Set the name of the columns"""  # noqa: DAR101,DAR201
        items = self._update_sizes(items)
        self._header = items

    def set_title(self, title: str):
        """Set the title (optional)"""  # noqa: DAR101,DAR201
        self._title = title

    def set_fmt(self, fmt: str):
        """Set format description (optional)"""  # noqa: DAR101,DAR201
        self._fmt = fmt

    def add_note(self, note: str):
        """Add a note (footer position)"""  # noqa: DAR101,DAR201
        self._notes.append(note)

    def add_row(self, items: Union[List[str], str]):
        """Add a row (list of item)"""  # noqa: DAR101,DAR201
        items = self._update_sizes(items)
        self._datas.append(items)

    def add_separator(self, value: str = '-'):
        """Add a separtor (line)"""  # noqa: DAR101,DAR201
        self._datas.append(value)

    def _update_sizes(self, items: Union[List[str], str]) -> List[str]:
        """Update the column sizes"""  # noqa: DAR101,DAR201,DAR401
        items = [items] if isinstance(items, str) else items
        if not self._sizes:
            self._sizes = [len(str(item)) + TableWriter.N_SPACE for item in items]
        else:
            if len(items) > len(self._sizes):
                raise ValueError('Size of the provided row is invalid')
            for i, item in enumerate(items):
                self._sizes[i] = max(len(str(item)) + TableWriter.N_SPACE, self._sizes[i])
        return items

    def _write_row(self, items: List[str], fmt):
        """Create a formated row"""  # noqa: DAR101,DAR201
        nfmt = ['.'] * len(self._sizes)
        for i, val in enumerate(fmt):
            if i < len(nfmt):
                nfmt[i] = val
        row = ''
        for i, item in enumerate(items):
            sup = self._sizes[i] - len(str(item))
            if nfmt[i] == '>':
                row += ' ' * sup + str(item) + ' ' * len(self._csep)
            else:
                row += str(item) + ' ' * sup + ' ' * len(self._csep)
        self.write(row)

    def _write_separator(self, val: str):
        """Create a formatted separator"""  # noqa: DAR101,DAR201
        row = ''
        for size in self._sizes:
            row += val * size + self._csep
        self.write(row)

    def write(self, msg: str, newline: str = '\n'):
        """Write fct"""  # noqa: DAR101,DAR201
        super(TableWriter, self).write(' ' * self._indent + msg + newline)

    def getvalue(self, fmt: str = '', endline: bool = False):
        """Buid and return the formatted table"""  # noqa: DAR101,DAR201

        fmt = fmt if fmt else self._fmt

        self.write('')
        if self._title:
            self.write(self._title)
            self._write_separator('-')
        if self._header:
            self._write_row(self._header, fmt)
            self._write_separator('-')
        for data in self._datas:
            if isinstance(data, str):
                self._write_separator(data)
            else:
                self._write_row(data, fmt)
        if endline or self._notes:
            self._write_separator('-')
        for note in self._notes:
            self.write(note)
        buff = super(TableWriter, self).getvalue()
        return buff

    def __str__(self):
        return self.getvalue()


def truncate_name(name: str, maxlen: int = 30):
    """Return a truncated string"""  # noqa: DAR101, DAR201
    maxlen = max(maxlen, 4)
    l_, r_ = (3, 1) if maxlen <= 12 else (12, 10)
    return (name[:maxlen - l_] + ".." + name[-r_:]) if maxlen < len(name) else name
