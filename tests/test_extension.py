import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd,geopandas as gpd
from shapely.geometry import LineString
import extended_experiment as e
class ExtensionTests(unittest.TestCase):
    def roads(self):
        g=gpd.GeoDataFrame({'road_class':['primary','primary','secondary']},geometry=[LineString([(0,0),(100,0)]),LineString([(100,0),(0,0)]),LineString([(0,20),(100,20)])],crs=32650)
        g,_=e.canonical_roads(g); g['endpoint_heading']=[e.core.legacy.line_heading(x) for x in g.geometry]; return g
    def test_geometry_identity(self):
        g=self.roads(); self.assertEqual(len(g),2)
        h,_=e.canonical_roads(g.iloc[::-1]); self.assertEqual(list(g.identity),list(h.identity))
    def test_user_separation(self):
        s=pd.DataFrame({'trip_id':[str(i) for i in range(60)],'source_user_id':[str(i//3) for i in range(60)]}); a,b=e.user_split(s)
        self.assertFalse(set(s[s.trip_id.isin(a)].source_user_id)&set(s[s.trip_id.isin(b)].source_user_id)); self.assertEqual(len(b),15)
    def test_features_no_ids_or_labels(self):
        g=self.roads(); xy=np.array([[10.,1.],[30.,1.],[50.,1.]]); targets=np.array([e.core.nearest(e.shapely.Point(p),g) for p in xy]); c,p=e.make_batch(xy,np.array([0.,5.,10.]),targets,g,'gaussian',0,42,1)
        for contextual in [False,True]:
            x=e.feature_frame(c,contextual); self.assertFalse({'candidate_id','point_id','is_target'}&set(x)); self.assertTrue(np.isfinite(x.to_numpy()).all())
        self.assertFalse(p.heading_available.iloc[0]); self.assertTrue(p.heading_available.iloc[1])
    def test_causal_context(self):
        g=self.roads(); a=np.array([[10.,1.],[30.,1.],[50.,1.]]); b=a.copy(); b[-1]=[90,90]; target=np.zeros(3,dtype=int)
        c,_=e.make_batch(a,np.arange(3)*5.,target,g,'gaussian',0,42,1); d,_=e.make_batch(b,np.arange(3)*5.,target,g,'gaussian',0,42,1)
        pd.testing.assert_frame_equal(c[c.point_id<2],d[d.point_id<2])
if __name__=='__main__': unittest.main()
