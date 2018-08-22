import os
import itertools
from collections import defaultdict
import numpy as np
from six import iteritems


class SparseDataset:
    def __init__(self, file_path, skip_lines=1, sep=",", rating_scale=(1, 5)):
        self.ratings_file = file_path
        self.sep = sep
        self.rating_scale = rating_scale
        self._global_mean = None
        with open(os.path.expanduser(self.ratings_file)) as f:
            self.raw_ratings = [self.parse_line(line) for line in
                           itertools.islice(f, skip_lines, None)]

    def parse_line(self, line, columns=4):
        line = line.split(self.sep)
        try:
            uid, iid, r, timestamp = (line[i].strip()
                                          for i in range(columns))
        except IndexError:
            raise ValueError('Impossible to parse line{0}. Check the line format and sep parameters.'.format(line))

        return uid, iid, float(r), timestamp

    def build_trainset(self):
        self.raw2inner_id_users = {}
        self.raw2inner_id_items = {}

        current_u_index = 0
        current_i_index = 0

        self.ur = defaultdict(list)
        self.ir = defaultdict(list)

        # user raw id, item raw id, translated rating, time stamp
        # inner id is continuous id starting from 0
        for urid, irid, r, timestamp in self.raw_ratings:
            try:
                uid = self.raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                self.raw2inner_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = self.raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                self.raw2inner_id_items[irid] = current_i_index
                current_i_index += 1

            self.ur[uid].append((iid, r))
            self.ir[iid].append((uid, r))

        self.n_users = len(self.ur)  # number of users
        self.n_items = len(self.ir)  # number of items
        self.n_ratings = len(self.raw_ratings)

        return self

    def all_ratings(self):
        """Generator function to iterate over all ratings.

        Yields:
            A tuple ``(uid, iid, rating)`` where ids are inner ids.
        """

        for u, u_ratings in iteritems(self.ur):
            for i, r in u_ratings:
                yield u, i, r

    @property
    def global_mean(self):
        """Return the mean of all ratings.

        It's only computed once."""
        if self._global_mean is None:
            self._global_mean = np.mean([r for (_, _, r) in
                                         self.all_ratings()])

        return self._global_mean

    def is_item_rated(self, iid):
        """Indicate if the item is rated at least once.

        Args:
            iid(int): The (inner) item id. 
        Returns:
            ``True`` if item is is rated at least once, else ``False``.
        """

        return iid in self.ir

    def is_user_rated(self, uid):
        """Indicate if the user has at least one rating.

        Args:
            uid(int): The (inner) user id. 
        Returns:
            ``True`` if the user has at least one rating, else ``False``.
        """

        return uid in self.ur

    def to_inner_uid(self, ruid):
        """Convert a **user** raw id to an inner id.

        Args:
            ruid(str): The user raw id.

        Returns:
            int: The user inner id.

        Raises:
            ValueError: When user is not part of the trainset.
        """

        try:
            return self.raw2inner_id_users[ruid]
        except KeyError:
            raise ValueError('User ' + str(ruid) +
                             ' is not part of the trainset.')

    def to_inner_iid(self, riid):
        """Convert an **item** raw id to an inner id.

        Args:
            riid(str): The item raw id.

        Returns:
            int: The item inner id.

        Raises:
            ValueError: When item is not part of the trainset.
        """

        try:
            return self.raw2inner_id_items[riid]
        except KeyError:
            raise ValueError('Item ' + str(riid) +
                             ' is not part of the trainset.')

