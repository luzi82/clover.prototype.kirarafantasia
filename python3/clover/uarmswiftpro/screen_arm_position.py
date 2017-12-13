import numpy as np
import json

class ScreenArmPosition:

    def __init__(self,filename,jjson=None):
        if filename != None:
            with open(filename,'r') as fin:
                jjson = json.load(fin)
        self.json = jjson
        self.cal_data = {}
        self.cal_data['screen_np']  = np.array([[u['screen_x'],u['screen_y']]    for u in jjson['point_list']])
        self.cal_data['screen_np1'] = np.array([[u['screen_x'],u['screen_y'],1.] for u in jjson['point_list']])
        self.cal_data['arm_np']     = np.array([[u['arm_x'],   u['arm_y']]       for u in jjson['point_list']])
        self.cal_data['arm_np1']    = np.array([[u['arm_x'],   u['arm_y'],   1.] for u in jjson['point_list']])

    def screen_to_arm(self, screen_x, screen_y):
        return self._point_convert(screen_x, screen_y, self.cal_data['screen_np'], self.cal_data['screen_np1'], self.cal_data['arm_np'])
        
    def arm_to_screen(self, arm_x, arm_y):
        return self._point_convert(arm_x, arm_y, self.cal_data['arm_np'], self.cal_data['arm_np1'], self.cal_data['screen_np'])
        
    def _point_convert(self, x, y, ffrom, ffrom1, tto):
        idx_list = self._select_three_point(np.array([x,y]), ffrom, tto)
        fffrom = np.take(ffrom1,idx_list,axis=0)
        ttto = np.take(tto,idx_list,axis=0)
        mat = np.linalg.solve(fffrom,ttto)
        arm_xy = np.dot(np.array([x,y,1]), mat)
        return float(arm_xy[0]),float(arm_xy[1])

    def _select_three_point(self, xy, ffrom, tto):
        delta = ffrom - xy
        distance = np.sum(delta*delta,axis=1)
        distance_sort_idx = np.argsort(distance)
        pt0 = ffrom[distance_sort_idx[0]]
        pt1 = ffrom[distance_sort_idx[1]]
        pt10 = pt1 - pt0
        pt10u = pt10/np.linalg.norm(pt10)
        for i in distance_sort_idx[2:]:
            ptt = ffrom[i]
            ptt0 = ptt - pt0
            ptt0u = ptt0/np.linalg.norm(ptt0)
            cross_abs = abs(float(np.cross(pt10u,ptt0u)))
            if cross_abs > 0.5:
                return [
                    distance_sort_idx[0],
                    distance_sort_idx[1],
                    i
                ]
        assert(False)
