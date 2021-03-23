# Unit test code for BDNE
# Author : Patrick Parkinson <patrick.parkinson@manchester.ac.uk>

import unittest
from BDNE import connect_big_query
from BDNE import data_structures


class TestDataStructures(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Set-up : connect via big_query"""
        connect_big_query()

    def test_wire(self):
        """General test for operation of Wire data_structure"""
        w = data_structures.Wire(1000)
        self.assertEqual(len(w.experiments()), 15)
        s = w.get('spectra_int')
        self.assertEqual(s, 73192500)

    def test_wire_collection(self):
        """General test for operation of WireCollection data_structure"""
        wc = data_structures.WireCollection()
        wc.load_sample(47)
        self.assertEqual(len(wc), 21162)


if __name__ == '__main__':
    unittest.main()
