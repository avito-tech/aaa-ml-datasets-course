from collections import namedtuple

MAX_IMG_SIZE = (1280, 960)
MAX_IMG_SIZE_STR = '1280x960'
NUM_CLASSES = 20
IMAGES_DIR = '/data/images/'
LABELS_PATH = '/app/data/labels.csv'


ROOM_TYPE = namedtuple('ROOM_TYPE', ['name', 'value_id', 'exclude'])

ROOM_TYPES = (
    ROOM_TYPE('ошибка загрузки изображения', -1, True),
    ROOM_TYPE('кухня / столовая', 0, False),
    ROOM_TYPE('кухня-гостиная', 1, False),
    ROOM_TYPE('универсальная комната', 2, False),
    ROOM_TYPE('гостиная', 3, False),
    ROOM_TYPE('спальня', 4, False),
    ROOM_TYPE('кабинет', 5, False),
    ROOM_TYPE('детская', 6, False),
    ROOM_TYPE('ванная комната', 7, False),
    ROOM_TYPE('туалет', 8, False),
    ROOM_TYPE('совмещенный санузел', 9, False),
    ROOM_TYPE('коридор / прихожая', 10, False),
    ROOM_TYPE('гардеробная / кладовая / постирочная', 11, False),
    ROOM_TYPE('балкон / лоджия', 12, False),
    ROOM_TYPE('вид из окна / с балкона', 13, False),
    ROOM_TYPE('дом снаружи / двор', 14, False),
    ROOM_TYPE('подъезд / лестничная площадка', 15, False),
    ROOM_TYPE('другое', 16, False),
    ROOM_TYPE('предметы интерьера / быт.техника', 17, False),
    ROOM_TYPE('не могу дать ответ / не ясно', 18, True),
    ROOM_TYPE ('комната без мебели', 19, False),
)


VALID_ROOM_TYPES = tuple(filter(lambda x: not x.exclude, ROOM_TYPES))

CLASS_NAME_MAPPING = {room_type.value_id: room_type.name for room_type in VALID_ROOM_TYPES}
