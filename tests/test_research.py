import sys, unittest, tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import research_experiment as r
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point,LineString
from sklearn.ensemble import RandomForestClassifier
class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.roads=gpd.GeoDataFrame(geometry=[LineString([(0,0),(10,0)]),LineString([(0,20),(10,20)]),LineString([(0,200),(10,200)])],crs=32650)
    def test_crs(self):
        x,y=r.TRANSFORM.transform(117,40); self.assertAlmostEqual(x,500000,places=4); self.assertTrue(4400000<y<4500000)
    def test_distance_and_nearest(self):
        self.assertEqual(r.nearest(Point(5,199),self.roads),2); self.assertAlmostEqual(self.roads.geometry.iloc[0].distance(Point(5,3)),3)
    def test_candidates_omission(self):
        ids,ds,radius=r.candidates(Point(5,1),self.roads,1); self.assertEqual(list(ids),[0]); self.assertNotIn(2,ids); self.assertEqual(radius,100); self.assertEqual(ds[0],1)
    def test_expansion(self):
        ids,ds,radius=r.candidates(Point(5,500),self.roads); self.assertEqual(radius,500); self.assertEqual(ids[0],2)
    def test_noise(self):
        xy=np.zeros((100000,2))
        for k in r.legacy.DEGRADATIONS: np.testing.assert_array_equal(r.legacy.perturb(xy[:30],k,10,42),r.legacy.perturb(xy[:30],k,10,42))
        e=r.legacy.perturb(xy,'gaussian',10,42); self.assertAlmostEqual(e.std(),10,delta=.1)
        e=r.legacy.perturb(xy[:30],'persistent_bias',10,42); np.testing.assert_allclose(np.linalg.norm(e,axis=1),10)
        e=r.legacy.perturb(xy,'correlated_drift',10,42); self.assertAlmostEqual(e.std(),10,delta=.3); self.assertAlmostEqual(np.corrcoef(e[:-1,0],e[1:,0])[0,1],.95,delta=.01)
    def test_split(self):
        a,b=r.split_ids([f'trip_{i:03}' for i in range(30)]); self.assertFalse(a&b); self.assertEqual((len(a),len(b)),(22,8))
    def test_metrics(self):
        d=pd.DataFrame(dict(target=[1,2,3],nearest=[1,1,1],present=[True,True,False],target_class=['a','b','b'],nearest_class=['a','a','a']))
        m=r.metrics(d,'nearest'); self.assertAlmostEqual(m['segment_e2e'],1/3); self.assertEqual(m['segment_conditional'],.5); self.assertAlmostEqual(m['candidate_recall'],2/3)
    def test_bad_input(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.csv'; p.write_text('latitude,longitude,timestamp\n91,2,2008-01-01\n')
            with self.assertRaises(ValueError): r.read_trip(p)
            p.write_text('latitude\n1\n')
            with self.assertRaises(ValueError): r.read_trip(p)
            p.write_text('latitude,longitude,timestamp\n39,116,invalid\n')
            with self.assertRaises(ValueError): r.read_trip(p)
    def test_prediction(self):
        model=RandomForestClassifier(n_estimators=3,random_state=42).fit([[0],[1],[2],[3]],[0,0,1,1]); p=model.predict_proba([[1],[2]]); self.assertEqual(p.shape,(2,2)); np.testing.assert_allclose(p.sum(axis=1),1)
if __name__=='__main__': unittest.main()
