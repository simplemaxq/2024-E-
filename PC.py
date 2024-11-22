import cv2
import numpy as np
import time
# import wiringpi

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

chess_data = [0, 0, 0, 0, 0, 0, 0, 0, 0]
chess_data_mem = [0, 0, 0, 0, 0, 0, 0, 0, 0]
chess_data_temp = [0, 0, 0, 0, 0, 0, 0, 0, 0]
white_chess_data = []
black_chess_data = []

mode = 0
status = 0
square = [2, 3, 4, 5]
play_chess_flag = 0



def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).squeeze(), (p2 - p1).squeeze()
    return abs(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))


def are_rectangles_close(center1, center2):
    threshold = 20  # 根据实际情况可能需要调整这个值
    return np.linalg.norm(center1 - center2) <= threshold


def sort_vertices(_approx):
    sort_vertices_center = _approx.mean(axis=0)

    def sort_criteria(points):
        return np.arctan2(points[0][1] - sort_vertices_center[0][1], points[0][0] - sort_vertices_center[0][0])
    sorted_vertices = sorted(_approx, key=sort_criteria)
    return np.array(sorted_vertices)


def find_rectangles(_img_canny):
    max_area = 0
    max_rect = None
    _sorted_centers = []
    centers_data = []
    rectangles_data = []
    contours, _ = cv2.findContours(_img_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # 找9个矩形
    for cnt in contours:
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(approx)

        if len(approx) == 4 and area > 1000:
            cosines = [angle_cos(approx[i], approx[(i + 1) % 4], approx[(i + 2) % 4]) for i in range(4)]
            if all(cos < 0.1 for cos in cosines):
                approx_sorted = sort_vertices(approx)
                center_of_rect = approx_sorted.mean(axis=0)[0]
                too_close = any(are_rectangles_close(center_of_rect, existing_center)
                                for existing_center in centers_data)
                if not too_close:
                    centers_data.append(center_of_rect)
                    rectangles_data.append(approx_sorted)
                    # 在原图上标出矩形
                    cv2.drawContours(img_contour, [approx_sorted], -1, (255, 255, 0), 1)

                if area > max_area:
                    max_area = area
                    max_rect = approx_sorted

    if max_rect is not None:
        cv2.drawContours(img_contour, [max_rect], -1, (0, 255, 0), 1)
        top_left, top_right, bottom_right, bottom_left = max_rect[0], max_rect[1], max_rect[2], max_rect[3]
        mid_top = (top_left + top_right) / 2
        mid_right = (top_right + bottom_right) / 2
        mid_bottom = (bottom_left + bottom_right) / 2
        mid_left = (top_left + bottom_left) / 2
        center = (top_left + bottom_right) / 2

        key_points = {
            '1': top_left,
            '2': mid_top,
            '3': top_right,
            '4': mid_left,
            '5': center,
            '6': mid_right,
            '7': bottom_left,
            '8': mid_bottom,
            '9': bottom_right
        }

        # 初始化一个字典来保存每个关键点的最近中心点和距离
        nearest_centers = {k: (None, float('inf')) for k in key_points}

        # 对每个中心点进行迭代，找到它们最近的关键点
        for center in centers_data:
            for _key, point in key_points.items():
                dist = np.linalg.norm(center - point)
                if dist < nearest_centers[_key][1]:
                    nearest_centers[_key] = (center, dist)

        # 从nearest_centers提取排序后的中心点
        _sorted_centers = [nearest_centers[str(i)][0] for i in range(1, 10) if nearest_centers[str(i)][0] is not None]

    # 标号并绘制中心点和编号
        for index, center in enumerate(_sorted_centers, start=1):
            cv2.circle(img_contour, tuple(center.astype(int)), 5, (0, 255, 0), -1)
            cv2.putText(img_contour, f'{index}', tuple(center.astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return _sorted_centers


def find_chess(_img_blur):
    # 在识别前清空数组
    global white_chess_data
    global black_chess_data
    chess_data = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    global chess_data_mem
    global chess_data_temp
    white_chess_data = []
    black_chess_data = []
    # for index in range(len(chess_data)):
    #     chess_data[index] = 0

    _circles = cv2.HoughCircles(_img_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                param1=30, param2=30, minRadius=20, maxRadius=50)
    # 如果检测到圆
    if _circles is not None:
        # 将圆的参数转换为整数
        _circles = np.uint16(np.around(_circles))

        for circle in _circles[0, :]:
            # 获取圆心坐标和半径
            center = (circle[0], circle[1])
            radius = circle[2]

            chess_in_board = 0

            # 在原图像上绘制圆心
            cv2.circle(img_contour, center, 1, (0, 100, 100), 3)
            # 在原图像上绘制圆周
            cv2.circle(img_contour, center, radius, (255, 0, 255), 3)
            mask = np.zeros_like(img_blurr)
            # 在掩模上绘制白色圆
            cv2.circle(mask, center, radius, 255, -1)
            # 应用掩模，只保留圆内的部分
            masked_image = cv2.bitwise_and(img_blurr, mask)
            # cv2.imshow("masked", masked_image)
            # 仅考虑圆内部的像素
            circle_pixels = masked_image[mask > 0]  # 仅选取掩模非零处的像素
            # 计算圆内部的白色和黑色像素数量
            white_pixels = np.sum(circle_pixels > 128)  # 圆内部灰度大于128的像素数量
            black_pixels = np.sum(circle_pixels <= 128)  # 圆内部灰度小于或等于128的像素数量
            # print(white_pixels, black_pixels)
            for index, sorted_center in enumerate(sorted_centers, start=1):
                # 判断棋子是否在棋盘内
                if ((((sorted_center[0] - center[0]) ** 2 + (sorted_center[1] - center[1]) ** 2) ** 0.5)
                        < radius):
                    cv2.circle(img_contour, center, 10, (0, 255, 255), -1)
                    chess_in_board = 1
                    # 判断圆内是主要是黑色还是白色
                    if white_pixels > black_pixels:
                        chess_data[index - 1] = 1
                        # print(index, "白棋")
                    else:
                        chess_data[index - 1] = 2
                        # print(index, "黑棋")
            if chess_in_board == 0:
                if white_pixels > black_pixels:
                    white_chess_data.append((circle[0], circle[1]))
                else:
                    black_chess_data.append((circle[0], circle[1]))
    return [chess_data, black_chess_data, white_chess_data]


def detect_color(img, x, y, radius=5):
    # 在识别前清空
    for index in range(len(chess_data)):
        chess_data[index] = 0
    for index, sorted_center in enumerate(sorted_centers, start=1):
        # 获取点周围的区域
        region = img[sorted_center[1]-radius:sorted_center[1]+radius+1, sorted_center[0]-radius:sorted_center[0]+radius+1]

        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # 定义颜色范围
        red_lower1 = np.array([0, 70, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 70, 50])
        red_upper2 = np.array([180, 255, 255])
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 20, 255])
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 50])

        # 创建掩码
        mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_white = cv2.inRange(hsv, white_lower, white_upper)
        mask_black = cv2.inRange(hsv, black_lower, black_upper)

        # 计算各颜色的像素比例
        total_pixels = region.size // 3
        red_pixels = cv2.countNonZero(mask_red)
        white_pixels = cv2.countNonZero(mask_white)
        black_pixels = cv2.countNonZero(mask_black)

        # 判断颜色
        if red_pixels > white_pixels and red_pixels > black_pixels:
            return "Red"
        elif white_pixels > red_pixels and white_pixels > black_pixels:
            return "White"
        elif black_pixels > red_pixels and black_pixels > white_pixels:
            return "Black"
        else:
            return "Unknown"


def serial_send_data(_status, _data=None):
    data = []
    if _data is not None:
        data_float = [str(num) for num in list(float(num) for num in _data)]
        print(data_float)
        data.append("#P1=")
        data.append(data_float[0])
        data.append("!")
        data.append("#P2=")
        data.append(data_float[1])
        data.append("!")
        # data.append("#P1=")
        # data.append(data_float[0])
        # data.append("!")
        # data.append("#P2=")
        # data.append(data_float[1])
        # data.append("!")
    data.append("#P3=")
    data.append(str(_status))
    data.append("!")

    # 这里写通信的发送
    # for value in data:
    #     # bytes_data = struct.pack('str', value)
    #     for byte in value:
    #         wiringpi.serialPutchar(serial, ord(byte)) # 这里写通讯的发送函数
    print(data)


def list1x9_to_3x3(lst):
    if len(lst) != 9:
        raise ValueError("Input list must have exactly 9 elements.")
    return [lst[i:i+3] for i in range(0, 9, 3)]


def list3x3_to_1x9(lst):
    if len(lst) != 3 or any(len(row) != 3 for row in lst):
        raise ValueError("Input list must be a 3x3 matrix.")
    return [element for row in lst for element in row]


def check_chess_num(lst):
    chess_in_board_num = 0
    if len(lst) != 9:
        raise ValueError("Input list must have exactly 9 elements.")
    for index in range(len(lst)):
        if lst[index] != 0:
            chess_in_board_num += 1
    return chess_in_board_num


def chess_calculate(s):
    global chess_data
    global chess_data_mem
    global chess_data_temp
    if mode == 4:
        for indexx in range(len(s)):
            for indexy in range(len(s[indexx])):
                if s[indexx][indexy] == 1:
                    s[indexx][indexy] = 2
                elif s[indexx][indexy] == 2:
                    s[indexx][indexy] = 1 

    breaktf = 0
    if (mode == 4 and check_chess_num(chess_data) == 1):

        if (s[1][1] == 0) :
            s[1][1] = 2
        else :
            s[0][2] =2

    else:
#连三
        if (s[1][1] == 1 and s[2][0] == 1 and s[0][2] == 2 and s[2][2] == 0): s[2][2] = 2
        elif (s[1][1] == 2 and s[1][0] == 1 and s[0][2] == 0):s[0][2] = 2  # 起手
        elif (s[0][0] == 2 and s[0][1] == 2 and s[0][2] == 0) : s[0][2] = 2# 横着第一行连线
        elif (s[0][0] == 2 and s[0][2] == 2 and s[0][1] == 0) : s[0][1] = 2# 横着第一行连线
        elif (s[0][2] == 2 and s[0][1] == 2 and s[0][0] == 0) : s[0][0] = 2# 横着第一行连线

        elif (s[1][0] == 2 and s[1][2] == 2 and s[1][1] == 0) :s[1][1] = 2#横着第二行连线
        elif (s[1][2] == 2 and s[1][1] == 2 and s[1][0] == 0) :s[1][0] = 2#横着第二行连线
        elif (s[1][0] == 2 and s[1][1] == 2 and s[1][2] == 0) :s[1][2] = 2#横着第二行连线

        elif (s[2][0] == 2 and s[2][1] == 2 and s[2][2] == 0) :s[2][2] = 2#横着第三行连线
        elif (s[2][0] == 2 and s[2][2] == 2 and s[2][1] == 0) :s[2][1] = 2#横着第三行连线
        elif (s[2][2] == 2 and s[2][1] == 2 and s[2][0] == 0): s[2][0] = 2 # 横着第三行连线

        elif (s[0][0] == 2 and s[2][0] == 2 and s[1][0] == 0): s[1][0] = 2#竖着着第一行连线
        elif (s[0][0] == 2 and s[1][0] == 2 and s[2][0] == 0) :s[2][0] = 2#竖着着第一行连线
        elif (s[2][0] == 2 and s[1][0] == 2 and s[0][0] == 0) :s[0][0] = 2#竖着着第一行连线

        elif (s[2][1] == 2 and s[1][1] == 2 and s[0][1] == 0): s[0][1] = 2  # 竖着着第二行连线
        elif (s[0][1] == 2 and s[2][1] == 2 and s[1][1] == 0): s[1][1] = 2  # 竖着着第二行连线
        elif (s[0][1] == 2 and s[1][1] == 2 and s[2][1] == 0): s[2][1] = 2  # 竖着着第二行连线

        elif (s[2][2] == 2 and s[1][2] == 2 and s[0][2] == 0): s[0][2] = 2  # 竖着第三行
        elif (s[2][2] == 2 and s[0][2] == 2 and s[1][2] == 0): s[1][2] = 2  # 竖着第三行
        elif (s[0][2] == 2 and s[1][2] == 2 and s[2][2] == 0) :s[2][2] = 2  #竖着第三行



        elif (s[0][0] == 2 and s[2][2] == 2 and s[1][1] == 0) :s[1][1] = 2#斜着
        elif (s[2][2] == 2 and s[1][1] == 2 and s[0][0] == 0) :s[0][0] = 2#斜着
        elif (s[0][0] == 2 and s[1][1] == 2 and s[2][2] == 0) :s[2][2] = 2#斜着



        elif (s[0][2] == 2 and s[2][0] == 2 and s[1][1] == 0) :s[1][1] = 2#到斜着
        elif (s[0][2] == 2 and s[1][1] == 2 and s[2][0] == 0) :s[2][0] = 2#到斜着
        elif (s[2][0] == 2 and s[1][1] == 2 and s[0][2] == 0):s[0][2] = 2 # 到斜着

        #堵三
        elif (s[0][0] == 1 and s[0][1] == 1 and s[0][2] == 0) :s[0][2] = 2#横着第一行
        elif (s[0][0] == 1 and s[0][2] == 1 and s[0][1] == 0) :s[0][1] = 2#横着第一行
        elif (s[0][2] == 1 and s[0][1] == 1 and s[0][0] == 0) :s[0][0] = 2#横着第一行


        elif (s[1][0] == 1 and s[1][2] == 1 and s[1][1] == 0):s[1][1] = 2  # 横着第二行
        elif (s[1][0] == 1 and s[1][1] == 1 and s[1][2] == 0): s[1][2] = 2  # 横着第二行
        elif (s[1][2] == 1 and s[1][1] == 1 and s[1][0] == 0): s[1][0] = 2  # 横着第二行

        elif (s[2][0] == 1 and s[2][1] == 1 and s[2][2] == 0) :s[2][2] = 2#横着第三行
        elif (s[2][0] == 1 and s[2][2] == 1 and s[2][1] == 0) :s[2][1] = 2#横着第三行
        elif (s[2][2] == 1 and s[2][1] == 1 and s[2][0] == 0) :s[2][0] = 2#横着第三行

        elif (s[0][0] == 1 and s[2][0] == 1 and s[1][0] == 0) :s[1][0] = 2#竖着第一行
        elif (s[0][0] == 1 and s[1][0] == 1 and s[2][0] == 0) :s[2][0] = 2#竖着第一行
        elif (s[2][0] == 1 and s[1][0] == 1 and s[0][0] == 0) :s[0][0] = 2#竖着第一行

        elif (s[0][1] == 1 and s[2][1] == 1 and s[1][1] == 0): s[1][1] = 2  # 竖着第二行
        elif (s[0][1] == 1 and s[1][1] == 1 and s[2][1] == 0):s[2][1] = 2  # 竖着第二行
        elif (s[2][1] == 1 and s[1][1] == 1 and s[0][1] == 0): s[0][1] = 2  # 竖着第二行

        elif (s[0][2] == 1 and s[1][2] == 1 and s[2][2] == 0) :s[2][2] = 2#竖着第三行
        elif (s[0][2] == 1 and s[2][2] == 1 and s[1][2] == 0) :s[1][2] = 2#竖着第三行
        elif (s[2][2] == 1 and s[1][2] == 1 and s[0][2] == 0) :s[0][2] = 2#竖着第三行


        elif (s[0][0] == 1 and s[2][2] == 1 and s[1][1] == 0) :s[1][1] = 2#斜着
        elif (s[0][0] == 1 and s[1][1] == 1 and s[2][2] == 0) :s[2][2] = 2#斜着
        elif (s[2][2] == 1 and s[1][1] == 1 and s[0][0] == 0) :s[0][0] = 2# 斜着

        elif (s[0][2] == 1 and s[2][0] == 1 and s[1][1] == 0) :s[1][1] = 2#到斜着
        elif (s[0][2] == 1 and s[1][1] == 1 and s[2][0] == 0) :s[2][0] = 2#到斜着
        elif (s[2][0] == 1 and s[1][1] == 1 and s[0][2] == 0): s[0][2] = 2  # 到斜着

        elif (s[0][0] == 1 and s[2][2] == 1 and s[1][1] == 2 and s[2][1] == 0) :s[2][1] = 2
        elif (s[0][2] == 1 and s[2][0] == 1 and s[1][1] == 2 and s[2][1] == 0) :s[2][1] = 2
        elif (s[0][0] == 1 and s[1][1] == 2 and s[0][2] == 0) :s[0][2] = 2
        elif (s[0][2] == 1 and s[1][1] == 2 and s[0][0] == 0) :s[0][0] = 2
        elif (s[2][0] == 1 and s[1][1] == 2 and s[2][2] == 0) :s[2][2] = 2
        elif (s[2][2] == 1 and s[1][1] == 2 and s[2][0] == 0) :s[2][0] = 2
        else:
            for x in range(3):
                for y in range(3):
                    if (s[x][y] == 0):
                        s[x][y] = 2
                        breaktf = 1
                        break
                if (breaktf == 1): break

    if mode == 4:
        for indexx in range(len(s)):
            for indexy in range(len(s[indexx])):
                if s[indexx][indexy] == 1:
                    s[indexx][indexy] = 2
                elif s[indexx][indexy] == 2:
                    s[indexx][indexy] = 1 
    return s



def check(s):

    if mode == 4:
        for indexx in range(len(s)):
            for indexy in range(len(s[indexx])):
                if s[indexx][indexy] == 1:
                    s[indexx][indexy] = 2
                elif s[indexx][indexy] == 2:
                    s[indexx][indexy] = 1 
    for x in range(3):#横着的三行
        if (s[x][0] == 1 and s[x][1] == 1 and s[x][2] == 1):

            print( "You win!!!" )
            tpcheckend = True
            return tpcheckend



    for x in range(3):#竖着的三行

        if (s[0][x] == 1 and s[1][x] == 1 and s[2][x] == 1):

            print( "You win!!!" )
            tpcheckend = True
            return tpcheckend



    for x in range(3):#横着的三行

        if (s[x][0] == 2 and s[x][1] == 2 and s[x][2] == 2):

            print( "You lose!" )
            tpcheckend = True
            return tpcheckend



    for x in range(3):#竖着的三行

        if (s[0][x] == 2 and s[1][x] == 2 and s[2][x] == 2):

            print( "You lose!" )
            tpcheckend = True
            return tpcheckend



    if (s[0][0] == 1 and s[1][1] == 1 and s[2][2] == 1):
        print( "You win!!!" )
        tpcheckend = True
        return tpcheckend
    elif (s[0][2] == 1 and s[1][1] == 1 and s[2][0] == 1):

        print( "You win!" )
        tpcheckend = True
        return tpcheckend

    elif (s[0][0] == 2 and s[1][1] == 2 and s[2][2] == 2):

        print( "You lose!" )
        tpcheckend = True
        return tpcheckend


    elif (s[0][2] == 2 and s[1][1] == 2 and s[2][0] == 2):

        print( "You lose!" )
        tpcheckend = True
        return tpcheckend

    else:
        count = 0
        for x in range(3):
            for y in range(3):
                if (s[x][y] == 1 or s[x][y] == 2):
                    count+=1
        if (count == 9):

            print( "Draw!")
            tpcheckend = True
            return tpcheckend
    tpcheckend = False
    return tpcheckend


# serial = wiringpi.serialOpen('/dev/ttyACM0', 115200)


while cap.isOpened():
    ret, frame = cap.read()
    img = frame
    img_contour = img

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blurr = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blurr, 50, 150)

    # sorted_centers = find_rectangles()
    # find_chess()
    height, width = frame.shape[:2]
    cv2.putText(img_contour, f"resolution:{width}*{height}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    # print(chess_data)
    # print(white_chess_data, black_chess_data)
    

    key = cv2.waitKey(1) & 0xFF
    if key == ord('/'):
        break
    elif key == ord('0'):
        mode = 0
    elif key == ord('1'):
        mode = 1
    elif key == ord('2'):
        mode = 2
    elif key == ord('3'):
        mode = 3
    elif key == ord('4'):
        mode = 4 

    if mode == 1:
        mode = 0
        status = 0
        serial_send_data(0)
        time.sleep(2)
        sorted_centers = find_rectangles(img_canny)
        [chess_data, black_chess_data, white_chess_data] = find_chess(img_blurr)
        cv2.imshow("frame", img_contour)
        status = 1
        serial_send_data(1, black_chess_data[0])
        time.sleep(3)
        status = 2   
        serial_send_data(2, sorted_centers[4])
        time.sleep(3)
        status = 0
        serial_send_data(0)
    elif mode == 2:
        mode = 0
        serial_send_data(0)
        time.sleep(2)
        sorted_centers = find_rectangles(img_canny)
        [chess_data, black_chess_data, white_chess_data] = find_chess(img_blurr)
        cv2.imshow("frame", img_contour)
        print("Enter first place:")
        which_square = int(input())
        serial_send_data(1, white_chess_data[0])
        time.sleep(3)
        serial_send_data(2, sorted_centers[which_square])
        time.sleep(3)
        cv2.imshow("frame", img_contour)
        print("Enter second place:")
        which_square = int(input())
        serial_send_data(1, white_chess_data[1])
        time.sleep(3)
        serial_send_data(2, sorted_centers[which_square])
        time.sleep(3)
        cv2.imshow("frame", img_contour)
        print("Enter third place:")
        which_square = int(input())
        serial_send_data(1, black_chess_data[0])
        time.sleep(3)
        serial_send_data(2, sorted_centers[which_square])
        time.sleep(3)
        cv2.imshow("frame", img_contour)
        print("Enter forth place:")
        which_square = int(input())
        serial_send_data(1, black_chess_data[1])
        time.sleep(3)
        serial_send_data(2, sorted_centers[which_square])
        time.sleep(3)
        serial_send_data(0)
    elif mode == 3:
        chess_from = 0 
        chess_to = 0

        cv2.imshow("frame", img_contour)
        # 下第一步棋
        if play_chess_flag == 0:
            chess_data_temp = [0, 0, 0, 0, 0, 0, 0, 0, 0]  #  empty
            serial_send_data(0)
            time.sleep(1)
            sorted_centers = find_rectangles(img_canny)  #  find center only this time
            [chess_data, black_chess_data, white_chess_data] = find_chess(img_blurr)
            cv2.imshow("frame", img_contour)
            serial_send_data(1, black_chess_data[0])
            time.sleep(3)
            serial_send_data(2, sorted_centers[4])
            time.sleep(3)
            serial_send_data(0)
            time.sleep(2)  
            play_chess_flag = 1
            serial_send_data(3)
            [chess_data_mem, black_chess_data, white_chess_data] = find_chess(img_blurr)
            print("first:", chess_data_mem)
        # 等待按键
        elif play_chess_flag == 1:
            # if wiringpi.OrangePi_get_gpio_mode(5) == 1:
            #    play_chess_flag = 2
            if key == ord('-'):
                play_chess_flag = 2
                # ret, frame = cap.read()
                # img = frame
                # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
                [chess_data, black_chess_data, white_chess_data] = find_chess(img_blurr)
                print("ggmem:", chess_data_mem, "chess:", chess_data)
        # 观察然后下棋
        elif play_chess_flag == 2:

            cv2.imshow("frame", img_contour)
            # 这次只看棋子不看棋盘
            if chess_data_mem != chess_data and (check_chess_num(chess_data_mem) == check_chess_num(chess_data)):          # 以下是人类作弊时的纠错代码
                print("1111111")
                # chess_data_temp = list3x3_to_1x9(chess_calculate(list1x9_to_3x3(chess_data)))
                print(chess_data_temp)
                for index in range(len(chess_data)):
                    if chess_data_mem[index] != chess_data[index] and chess_data[index] == 0:
                        chess_to = index
                    elif chess_data_mem[index] != chess_data[index] and chess_data_mem[index] == 0:
                        chess_from = index
                serial_send_data(1, sorted_centers[chess_from])
                time.sleep(3)
                serial_send_data(2, sorted_centers[chess_to])
                time.sleep(3)
                serial_send_data(0)
                time.sleep(2)
                play_chess_flag = 1
                # chess_data=find_chess()
                cv2.imshow("frame", img_contour)
                serial_send_data(3)
                [chess_data_mem, black_chess_data, white_chess_data] = find_chess(img_blurr)               
            elif chess_data_mem != chess_data and (check_chess_num(chess_data_mem) != check_chess_num(chess_data)):
                print("222222")
                chess_data_temp = list3x3_to_1x9(chess_calculate(list1x9_to_3x3(chess_data)))
                print(chess_data_temp)
                for index in range(len(chess_data_temp)):
                    if chess_data[index] != chess_data_temp[index] and chess_data[index] == 0:
                        chess_to = index
                        print("chess_to", chess_to)
                serial_send_data(1, black_chess_data[0])
                time.sleep(3)
                serial_send_data(2, sorted_centers[chess_to])
                time.sleep(3)
                serial_send_data(0)
                print("mem:", chess_data_mem, "chess:", chess_data)
                time.sleep(2)
                play_chess_flag = 1
                ret, frame = cap.read()
                img = frame
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
                # chess_data=find_chess()
                cv2.imshow("frame", img_contour)
                print("mem:", chess_data_mem, "chess:", chess_data)
                serial_send_data(3)
                chess_data_mem = chess_data_temp
                print("mem:", chess_data_mem, "chess:", chess_data)
                
            elif chess_data_mem == chess_data:
                # 这里暂时先这么写，回头处理如果人没有下棋但是按了按键
                print("333333")
                chess_data_temp = list3x3_to_1x9(chess_calculate(list1x9_to_3x3(chess_data)))

                for index in range(len(chess_data_temp)):
                    if chess_data[index] != chess_data_temp[index] and chess_data[index] == 0:
                        chess_to = index
                        print("chess_to", chess_to)
                serial_send_data(1, black_chess_data[0])
                time.sleep(3)
                serial_send_data(2, sorted_centers[chess_to])
                time.sleep(3)
                serial_send_data(0)
                time.sleep(2)
                play_chess_flag = 1
                ret, frame = cap.read()
                img = frame
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
                # chess_data=find_chess()
                cv2.imshow("frame", img_contour)
                print("mem:", chess_data_mem, "chess:", chess_data)
                serial_send_data(3)
                chess_data_mem = chess_data_temp
                print("mem:", chess_data_mem, "chess:", chess_data)
        if (check(list1x9_to_3x3(chess_data)) is True) or (check(list1x9_to_3x3(chess_data_temp)) is True):
            mode = 0  #游戏结束
            play_chess_flag = 0
            
    elif mode == 4:
        chess_from = 0 
        chess_to = 0

        cv2.imshow("frame", img_contour)
        # 下第一步棋
        if play_chess_flag == 0:
            chess_data_temp = [0, 0, 0, 0, 0, 0, 0, 0, 0]  #  empty
            serial_send_data(0)
            time.sleep(1)
            sorted_centers = find_rectangles(img_canny)  #  find center only this time
            [chess_data, black_chess_data, white_chess_data] = find_chess(img_blurr)
            cv2.imshow("frame", img_contour)
            play_chess_flag = 1
            chess_data_mem = chess_data
        # 等待按键
        elif play_chess_flag == 1:
            # if wiringpi.OrangePi_get_gpio_mode(5) == 1:
            #    play_chess_flag = 2
            if key == ord('-'):
                play_chess_flag = 2
                [chess_data, black_chess_data, white_chess_data] = find_chess(img_blurr)
        # 观察然后下棋
        elif play_chess_flag == 2:
            cv2.imshow("frame", img_contour)
            # 这次只看棋子不看棋盘
            print("mem:", chess_data_mem, "chess:", chess_data)
            if chess_data_mem != chess_data and (check_chess_num(chess_data_mem) == check_chess_num(chess_data)):          # 以下是人类作弊时的纠错代码

                print("!                ATTEENTION!!!!")
                # chess_data_temp = list3x3_to_1x9(chess_calculate(list1x9_to_3x3(chess_data)))
                print(chess_data_temp)
                for index in range(len(chess_data)):
                    if chess_data_mem[index] != chess_data[index] and chess_data[index] == 0:
                        chess_to = index
                    elif chess_data_mem[index] != chess_data[index] and chess_data_mem[index] == 0:
                        chess_from = index
                serial_send_data(1, sorted_centers[chess_from])
                time.sleep(3)
                serial_send_data(2, sorted_centers[chess_to])
                time.sleep(3)
                serial_send_data(0)
                time.sleep(2)
                play_chess_flag = 1 
                cv2.imshow("frame", img_contour)
                serial_send_data(3)
                [chess_data_mem, black_chess_data, white_chess_data] = find_chess(img_blurr)               
            elif chess_data_mem != chess_data and (check_chess_num(chess_data_mem) != check_chess_num(chess_data)):
                print("!                ATTEENTION!!!!")
                chess_data_temp = list3x3_to_1x9(chess_calculate(list1x9_to_3x3(chess_data)))
                print(chess_data_temp)
                for index in range(len(chess_data_temp)):
                    if chess_data[index] != chess_data_temp[index] and chess_data[index] == 0:
                        chess_to = index
                        print("chess_to", chess_to)
                serial_send_data(1, white_chess_data[0])
                time.sleep(3)
                serial_send_data(2, sorted_centers[chess_to])
                time.sleep(3)
                serial_send_data(0)
                time.sleep(2)
                play_chess_flag = 1
                [chess_data, black_chess_data, white_chess_data] = find_chess(img_blurr)
                cv2.imshow("frame", img_contour)
                serial_send_data(3)
                chess_data_mem = chess_data_temp
            elif chess_data_mem == chess_data:
                # 这里暂时先这么写，回头处理如果人没有下棋但是按了按键
                chess_data_temp = list3x3_to_1x9(chess_calculate(list1x9_to_3x3(chess_data)))
                print(chess_data_temp)
                for index in range(len(chess_data_temp)):
                    if chess_data[index] != chess_data_temp[index] and chess_data[index] == 0:
                        chess_to = index
                        print("chess_to", chess_to)
                serial_send_data(1, white_chess_data[0])
                time.sleep(3)
                serial_send_data(2, sorted_centers[chess_to])
                time.sleep(3)
                serial_send_data(0)
                time.sleep(2)
                play_chess_flag = 1
                [chess_data, black_chess_data, white_chess_data] = find_chess(img_blurr)
                cv2.imshow("frame", img_contour)
                serial_send_data(3)
                chess_data_mem = chess_data_temp
        if (check(list1x9_to_3x3(chess_data)) is True) or (check(list1x9_to_3x3(chess_data_temp)) is True):
            mode = 0  #游戏结束
            play_chess_flag = 0
          
    else:
        sorted_centers = find_rectangles(img_canny)
        [chess_data, black_chess_data, white_chess_data] = find_chess(img_blurr)
        cv2.imshow("frame", img_contour)

cap.release()
cv2.destroyAllWindows()
