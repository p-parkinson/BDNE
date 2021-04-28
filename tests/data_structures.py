# Unit test code for BDNE
# Author : Patrick Parkinson <patrick.parkinson@manchester.ac.uk>

import unittest
from BDNE import connect_mysql
from BDNE import data_structures
from pandas import DataFrame
from numpy import ndarray


class TestDataStructures(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Set-up : connect via my_sql at the start of the unittest"""
        connect_mysql()

    def test_entity(self):
        """General test for operation of Wire data_structure"""
        w = data_structures.Entity(1000)
        # Check experiments
        self.assertEqual(len(w.experiments()), 15)
        s = w.get('spectra_int')
        # Check a single parameter
        self.assertEqual(s, 73192500)

    def test_entity_collection(self):
        """General test for operation of EntityCollection data_structure"""
        wc = data_structures.EntityCollection()
        wc.load_sample(47)
        # Check length
        self.assertEqual(len(wc), 21162)
        # Get a wire and test
        w = wc.get_entity(1)
        self.assertIs(type(w), data_structures.Entity)
        self.assertEqual(w.db_id, 84439)

    def test_measurement_collection(self):
        """General test for a MeasurementCollection"""
        wc = data_structures.EntityCollection()
        wc.load_sample(47)
        mc = wc.get_measurement('l')
        # Test number
        self.assertEqual(len(mc),21162)
        # Test type
        self.assertIs(type(mc),data_structures.MeasurementCollection)
        # Test matrix collect
        m = mc.collect_as_matrix()
        self.assertAlmostEqual(sum(m), 269038, delta = 1)
        # Test sample output type
        o = mc.sample(1)
        self.assertIs(type(o),DataFrame)

    def test_postprocess(self):
        """Testing the postprocess class"""
        wc = data_structures.EntityCollection()
        wc.load_sample(47)
        mc = wc.get_measurement('l')
        pp = data_structures.PostProcess()
        pp.set_data(mc)
        pp.set_function(lambda x: x**2)
        # Collect
        pp_out = pp.collect_as_matrix()
        mc_out = mc.collect_as_matrix()
        # Check output type
        self.assertIs(type(pp_out),ndarray)
        # Check function
        self.assertEqual(pp_out[0],mc_out[0]**2)


if __name__ == '__main__':
    unittest.main()
