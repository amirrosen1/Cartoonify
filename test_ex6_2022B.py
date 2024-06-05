from cartoonify import *
from ex6_helper import *


def test_separate_channels():
    assert separate_channels([[[1, 2]]]) == [[[1]], [[2]]]
    assert separate_channels([[[1, 2], [1, 2]]]) == [[[1, 1]], [[2, 2]]]
    assert separate_channels([[[1, 2]], [[1, 2]]]) == [[[1], [1]], [[2], [2]]]
    assert separate_channels([[[1, 2, 3]]]) == [[[1]], [[2]], [[3]]]
    assert separate_channels([[[1, 2], [3, 4]]]) == [[[1, 3]], [[2, 4]]]
    assert separate_channels([[[1, 2]], [[3, 4]]]) == [[[1], [3]], [[2], [4]]]
    assert separate_channels(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]) == [
               [[1, 4], [7, 10]], [[2, 5], [8, 11]], [[3, 6], [9, 12]]]
    assert separate_channels([[[1, 2, 3]]*3]*4) == [[[1]*3]*4, [[2]*3]*4, [[3]*3]*4]


def test_combine_channels():
    assert combine_channels([[[1]], [[2]], [[3]]]) == [[[1, 2, 3]]]
    assert combine_channels([[[1, 3]], [[2, 4]]]) == [[[1, 2], [3, 4]]]
    assert combine_channels([[[1]], [[2]]]) == [[[1, 2]]]
    assert combine_channels([[[1, 1]], [[2, 2]]]) == [[[1, 2], [1, 2]]]
    assert combine_channels([[[1], [1]], [[2], [2]]]) == [[[1, 2]], [[1, 2]]]
    assert combine_channels(
        [[[1, 4], [7, 10]], [[2, 5], [8, 11]], [[3, 6], [9, 12]]]) == [
               [[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    assert combine_channels([[[1]*3]*4, [[2]*3]*4, [[3]*3]*4]) == [[[1, 2, 3]]*3]*4


def test_RGB2grayscale():
    assert RGB2grayscale([[[100, 180, 240]]]) == [[163]]
    assert RGB2grayscale([[[100, 180, 240], [100, 180, 240]],
                          [[100, 180, 240], [100, 180, 240]]]) == [[163, 163],
                                                                   [163, 163]]
    assert RGB2grayscale([[[200, 0, 14], [15, 6, 50]]]) == [[61, 14]]

    # שאלתי בפורום אם רלוונטי - אבי הלוי
    # assert RGB2grayscale(
    #     [[[100, 180, 240], [1, 1, 1]], [[0, 0, 0], [-1, -2, 5]]]) == [[163, 1],
    #                                                                   [0, 0]]

# הקרנל הספציפי שהתבקשנו לממש כאן הוא קרנל שממצע את סביבתו, זה יעזור לנו בהמשך.
# מימוש פונקציית אפליי הקרנל עצמה:
# נצטרך משתנה שיגיד לנו מה גודל השוליים של הקרנל - נקבל אותו ע"י אורך הקרנל\\2
# בקרנל 3X3 נקבל גודל שוליים של 1 (מסביב לפיקסל עצמו שעליו מחושב הקרנל)
# נרצה לרוץ על אינדקסי התמונה ועל אינדסקי הקרנל עצמו ולהכפיל פיקסל בפיקסל
# שימו לב שכדי לרוץ על אינדקסי הקרנל, נרוץ על ריינג' בין שוליים*1- לשוליים+1
# נצטרך להחריג את המקרים בהם הקרנל מושם על פינות וצריך להשתמש בערכי פיקסל שבעצם לא קיימים
# נבין מתי זה קורה - כשמיקום האינדקס (אינדקס תמונה + אינדקס קרנל) קטן מאפס או >= לאורך השורה\טור
#במקרים אלה, נגיד לפונקציה לרוץ על מה שנמצא במיקום אינדקס התמונה שעליו אנחנו מפעילים את הקרנל כרגע.
# שימו לב שיש להתייחס למקרי קצה בהם ערכי פיקסל גדולים מ255 (ואז נשים בהם את 255) או קטנים מ0 (ואז נשים בהם 0)

def test_kernel():
    assert blur_kernel(3) == [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9],
                              [1 / 9, 1 / 9, 1 / 9]]
    assert blur_kernel(11) == [[1 / 121] * 11] * 11
    assert blur_kernel(1) == [[1.0]]
    assert apply_kernel([[0, 128, 255]], blur_kernel(3)) == [[14, 128, 241]]
    assert apply_kernel([[0, 128, 255]], blur_kernel(3)) == [[14, 128, 241]]
    assert apply_kernel([[1, 1, 5],
                         [7, 1, 0],
                         [3, 3, 3]], blur_kernel(3)) == [[2, 2, 4],
                                                         [4, 3, 1],
                                                         [3, 3, 2]]
    assert apply_kernel([[0]], blur_kernel(1)) == [[0]]
    assert apply_kernel([[255]], blur_kernel(1)) == [[255]]
    assert apply_kernel([[10, 20, 30, 40, 50], [8, 16, 24, 32, 40], [6, 12, 18, 24, 30],
[4, 8, 12, 16, 20]] , blur_kernel(5)) == [[12, 20, 26, 34, 44], [11, 17, 22, 27,
34], [10, 16, 20, 24, 29], [7, 11, 16, 18, 21]]

# ערכי הקוארדינטות המוזנים לפונקציה יכולים להיות כל ערך שנמצא בטווח התמונה
# העיקר הוא למצוא באיזו קוארדינטה ספציפית נמצאת הקוארדינטה
# לדוגמא, יהי תמונה בגודל 7X8 , קיבלנו קוארדינטות 5.7X3.1.
# נצטרך להבין איפה בדיוק הן נמצאות - בגובה בין 5 ל6 וברוחב בין 3 ל4 ואז נתייחס לקוארדינטות אלה כקוארדינטות של a,b,c,d
# של נוסחת האינטרפולציה וכך נגדיר גם את a,b,c,d
# שימו לב שקיבלנו אותן ע"י עיגול מעלה ומטה של שתי הערכים שהושמו לפונקציה.
# כדי להתאים לנוסחת האינטרפולציה נרצה לחסר 6-5.7 ו4-3.1 וכך לקבל ערכים עשרוניים איתם ניתן לעבוד.
def test_bilinear_interpolation():
    assert bilinear_interpolation([[0, 64], [128, 255]], 0, 0) == 0
    assert bilinear_interpolation([[0, 64], [128, 255]], 1, 1) == 255
    assert bilinear_interpolation([[0, 64], [128, 255]], 0.5, 0.5) == 112
    assert bilinear_interpolation([[0, 64], [128, 255]], 0.5, 1) == 160
    assert bilinear_interpolation([[0, 64], [128, 255]], 0, 1) == 64
    assert bilinear_interpolation([[0, 64], [128, 255]], 1, 1) == 255
    assert bilinear_interpolation([[0, 64], [128, 255]], 0.5, 0.5) == 112


#העיקר בפונקציה הזאת היא שפינות ממופות לפינות,
#  כלומר i==0 ו j==0 של התמונה החדשה היא i==0 וj==0 של התמונה הישנה וכן הלאה לכל 4 הפינות.
# את שאר הפיקסלים נמפה בתי לולאות שרצות על טווח אורכי מימדי התמונה החדשה ונשלח אותם לפונקציית הבייליניאר.
# חשוב! מכיוון שאנו רצים על לולאות ואינדקסים, יש להוריד 1 מערכי האורך והרוחב של התמונה החדשה וגם הישנה ורק אז לחשב את היחס בין
# האורך החדש לאורך ישן והרוחב החדש והרוחב הישן
# את היחסים נכפיל בi\j בתוך הלולאה השנייה ואת שתי המכפלות נזין בבייליניאר.
def test_resize():
    assert resize([[0, 1], [2, 3]], 10, 10)[9][9] == 3
    assert resize([[0, 1], [2, 3]], 10, 10)[0][0] == 0
    assert resize([[0, 1], [2, 3]], 10, 10)[0][9] == 1
    assert resize([[0, 1], [2, 3]], 10, 10)[9][0] == 2

#שימו לב שיש להבדיל בין תמונה צבעונית שהיא רשימה תלת מימדית ותמונה בשחור לבן שהיא רשימה דו מימדית
# ניתן להשתמש בפונקציית הפרדת הערוצים שמימשנו מקודם וב[1-::]
def test_rotate_90():
    assert rotate_90([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'R') == [[7, 4, 1],
                                                                 [8, 5, 2],
                                                                 [9, 6, 3]]
    assert rotate_90([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'L') == [[3, 6, 9],
                                                                 [2, 5, 8],
                                                                 [1, 4, 7]]
    assert rotate_90([[1, 2, 3], [4, 5, 6]], 'R') == [[4, 1],
                                                      [5, 2],
                                                      [6, 3]]
    assert rotate_90([[1, 2, 3], [4, 5, 6]], 'L') == [[3, 6],
                                                      [2, 5],
                                                      [1, 4]]
    assert rotate_90(
        [[[1, 2, 3], [2, 3, 4], [3, 4, 5]], [[4, 5, 6], [5, 6, 7], [6, 7, 8]]],
        'L') == [
               [[3, 4, 5], [6, 7, 8]],
               [[2, 3, 4], [5, 6, 7]],
               [[1, 2, 3], [4, 5, 6]]]
    assert rotate_90([[[1, 2, 3], [4, 5, 6]], [[0, 5, 9], [255, 200, 7]]], 'L') == [[[4, 5, 6], [255, 200, 7]], [[1, 2, 3], [0, 5, 9]]]

# בפונקציה זו בעצם נשתמש באפליי קרנל שבנינו פעמיים.
# פעם אחת עם קרנל שנוצר מבלוק סייז ופעם אחת עם קרנל שנוצר מבלר סייז
# נקבל שתי תמונות שאחת היא טשטוש ע"י קרנל של תמונת המקור והשנייה היא טשטוש ע"י קרנל של הראשונה ונשווה ביניהן לפי הנוסחא
# ולפי מה שיוצא נחליף את הפיקסל בפיקסל שכולו שחור או כולו לבן.
def test_get_edges():
    assert get_edges([[200, 50, 200]], 3, 3, 10) == [[255, 0, 255]]
    assert get_edges([[200, 50, 200], [200, 50, 200], [200, 50, 200]], 1, 3,
                     10) == [[255, 0, 255], [255, 0, 255], [255, 0, 255]]


def test_quantize():
    assert quantize([[0, 50, 100], [150, 200, 250]], 8) == [[0, 36, 109], [146, 219, 255]]

# שימו לב - המסיכה היא רשימה דו מימדית אך התמונות יכולות להיות תלת מימדיות (צבעוניות)
#פרקו למקרים כמו בפונקציית הרוטייט.
def test_mask():
    assert add_mask([[50, 50, 50]], [[200, 200, 200]], [[0, 0.5, 1]]) == [
        [200, 125, 50]]
    assert add_mask([[[1,2,3], [4,5,6]],[[7,8,9],[10,11,12]]], [[[250,250,250],
[0,0,0]],[[250,250,100],[1,11,13]]], [[0, 0.5, 1]]*2) == [[[250, 250, 250],
[2, 2, 3]], [[250, 250, 100], [6, 11, 12]]]
    assert add_mask([[50, 50, 50]], [[200, 200, 200]], [[0, 0.5, 1]]) == [[200, 125, 50]]


#פונקציית הקרטוניפיי:
# בשבילה נשתמש בכל הפונקציות שמימשנו עד עכשיו. איך זה עובד?
# נפעיל על התמונה המקורית:
# פונקציית ההאפרה שהופכת תמונה צבעונית שהיא רשימה תלת מימדית לתמונה אפורה שהיא רשימה דו מימדית.
# על התוצאה נפעיל את פונקציית גט אדג'ס שנותנת לנו את קווי המתאר של התמונה
# מהתמונה של קווי המתאר ניצור מאסק - נחלק כל פיקסל ב255 כדי לקבל 0\1
# נפעיל על התמונה המקורית את הפוקנציה שמרדדת ערוצי צבע בתמונה צבעונית
#נפעיל את פונקציית הפרדת על הערוצים של התמונה הצבעונית המרודדת
# על כל ערוץ צבע נפעיל מאסק שרצה על ערוץ הצבע, תמונת קווי המתאר והמאסק שיצרנו
# לבסוף נאחד הכל לתמונה אחת
# ויאללה ביי


def test_image_processing():
    image = load_image("ziggy.png")
    show_image(cartoonify(scale_down_colored_image(image, 460), 5, 15, 17, 8))
