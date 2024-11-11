def loadDanceDatabase():
    # 추후 파일 읽기로 구현
    ########## 이 부분을 가흔이의 정답 DB 읽어오는 코드로 변경 ##########

    normalized_answer_2d = [
        (190, 167),
        (190, 156),
        (191, 154),
        (192, 153),
        (186, 160),
        (185, 161),
        (184, 162),
        (195, 147),
        (183, 163),
        (193, 170),
        (190, 175),
        (211, 132),
        (184, 204),
        (219, 87),
        (155, 229),
        (226, 48),
        (129, 239),
        (231, 35),
        (121, 248),
        (231, 38),
        (120, 240),
        (230, 45),
        (123, 235),
        (261, 239),
        (239, 261),
        (284, 327),
        (216, 341),
        (299, 415),
        (188, 434),
        (298, 429),
        (186, 448),
        (308, 440),
        (187, 460),
    ]
    converted_answer_landmarks_3d = [
        (-0.43727564811706543, -0.30032679438591003, -0.3849524259567261),
        (-0.43142667412757874, -0.33429262042045593, -0.3867550194263458),
        (-0.4330572485923767, -0.33410361409187317, -0.37716224789619446),
        (-0.4336633086204529, -0.33270519971847534, -0.37859484553337097),
        (-0.45972687005996704, -0.3263115882873535, -0.3940877914428711),
        (-0.45783117413520813, -0.32696041464805603, -0.404888391494751),
        (-0.44522953033447266, -0.3206246793270111, -0.3844701051712036),
        (-0.3833053410053253, -0.4102371335029602, -0.29024016857147217),
        (-0.45418399572372437, -0.28523963689804077, -0.26014238595962524),
        (-0.405250608921051, -0.3307318389415741, -0.3364866375923157),
        (-0.4087167978286743, -0.2870800197124481, -0.35777777433395386),
        (-0.27067968249320984, -0.452536940574646, -0.14668165147304535),
        (-0.5084015130996704, -0.23653052747249603, -0.08049526810646057),
        (-0.22994059324264526, -0.6116040945053101, -0.12489988654851913),
        (-0.6304511427879333, -0.07821023464202881, -0.16523227095603943),
        (-0.1426747441291809, -0.7355309724807739, -0.09834575653076172),
        (-0.8222062587738037, -0.023136764764785767, -0.3212057650089264),
        (-0.13162779808044434, -0.7668865919113159, -0.08871200680732727),
        (-0.8585522174835205, -0.037090003490448, -0.3798617422580719),
        (-0.1375337541103363, -0.7552145719528198, -0.11958429962396622),
        (-0.8597759008407593, -0.09607228636741638, -0.3915078938007355),
        (-0.11980196833610535, -0.7282341718673706, -0.11400052905082703),
        (-0.8303636312484741, -0.03959214687347412, -0.3375690281391144),
        (0.07601739466190338, -0.05146893113851547, 0.011556041426956654),
        (-0.08021347224712372, 0.04621221125125885, -0.007473418023437262),
        (0.2741943597793579, 0.28189417719841003, -0.05302029103040695),
        (-0.30474597215652466, 0.2676483988761902, -0.06611897796392441),
        (0.42376476526260376, 0.6333851218223572, 0.10239139199256897),
        (-0.43576276302337646, 0.7075188755989075, 0.03948541730642319),
        (0.42394018173217773, 0.686928391456604, 0.12386489659547806),
        (-0.46053412556648254, 0.7477656602859497, 0.012583951465785503),
        (0.44388845562934875, 0.724605917930603, 0.05905136093497276),
        (-0.5732921361923218, 0.7279732823371887, -0.05549197643995285),
    ]

    return {
        0.1: (normalized_answer_2d, converted_answer_landmarks_3d),
        0.2: (normalized_answer_2d, converted_answer_landmarks_3d),
        0.3: (normalized_answer_2d, converted_answer_landmarks_3d),
        0.4: (normalized_answer_2d, converted_answer_landmarks_3d),
        0.5: (normalized_answer_2d, converted_answer_landmarks_3d),
        0.6: (normalized_answer_2d, converted_answer_landmarks_3d),
        0.7: (normalized_answer_2d, converted_answer_landmarks_3d),
        0.8: (normalized_answer_2d, converted_answer_landmarks_3d),
        0.9: (normalized_answer_2d, converted_answer_landmarks_3d),
        1.0: (normalized_answer_2d, converted_answer_landmarks_3d),
        1.1: (normalized_answer_2d, converted_answer_landmarks_3d),
        1.2: (normalized_answer_2d, converted_answer_landmarks_3d),
    }


def isKeyPointTime(current_time):
    # 추후 구현
    return True


def read_answer(database, current_time) -> tuple[list, list]:
    # 추후 구현
    print("시간: ", current_time, "s", sep="")

    # idx = current_time # 아직 DB 미완성
    idx = 0.1

    return database[idx]
