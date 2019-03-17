# Define Algorithms
from sympy import Line, Point, geometry


def find_intersection(ar_1, ar_2):
    lines_1 = [geometry.Polygon(Point(i, ar_1[i]), Point(i + 1, ar_1[i + 1])) for i in range(len(ar_1) - 1)]
    lines_2 = [geometry.Polygon(Point(i, ar_1[i]), Point(i + 1, ar_2[i + 1])) for i in range(len(ar_2) - 1)]

    for li in range(len(lines_1)):
        # print(lines_1[li].equation(), lines_2[li].equation())
        intersection = geometry.intersection(lines_1[li], lines_2[li])
        print(intersection)


def detect_2_min_trends(points, sma_1, sma_2, sma_3, macd_val, macd_signal):
    lasted_val = points[-1]

    print(sma_1)
    find_intersection(sma_1, sma_2)

    # Hypo
    '''
    if 
        sma1 > sma2 
            and 
        sma_3 < sma1, sma2 
    
      then this is a up trend    
    '''
