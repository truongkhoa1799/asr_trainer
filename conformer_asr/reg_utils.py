import re
from enum import Enum
from pathlib import Path
from vietnam_number import n2w

'''
Regular Expression for cleaning data
'''
class RegType(Enum):
    DATE = 0
    TIME = 1
    NUMBER = 2
    CURRENCY = 3
    
units_dir = Path('/home/khoa/NovaIntechs/src/Smart-Speaker-Common/smart_speaker_common/scripts/corpus/data')
unit_paths = list(units_dir.glob('*.tsv'))
units_dict = dict()
for unit_path in unit_paths:
    with open(unit_path, 'r') as fin:
        for line in fin.readlines():
            line = line.strip().lower().split('\t')
            units_dict[f"\{line[0]}" if line[0] in ["$"] else line[0]] = line[-1]

begin_pos = '(^| )'
end_pos = '($| |\. |\.$)'
chars_to_ignore_regex_begin = '[\?\!\;\"\'\(\)\{\}\“\‘\”\…]'  # remove special character tokens
chars_to_ignore_regex_end = '[\,\?\.\!\;\:\"\'\(\)\{\}\“\‘\”\…\-]'  # remove special character tokens
set_day_vocab = set(['ngày', 'đêm', 'hôm', 'hôm qua', 'sáng', 'trưa', 'tối', 'chiều', 'khuya', 'mùng'])

# ---------------------------------------- PATTERN REGEXP ----------------------------------------
date_patterrn_layer_1 = {
    "dd_mm_yyyy": f'''{begin_pos}(ngày|đêm|hôm|sáng|trưa|tối|chiều|khuya|mùng|hôm qua|hôm nay)( +)(\
([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|\-|\/)([1-9]|0[1-9]|1[0-2])(\.|\-|\/)([1-2][0-9][0-9][0-9])|\
([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|\-|\/)([1-9]|0[1-9]|1[0-2])|\
([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])\
){end_pos}''',

    "dd_mm_yyyy_1": f'''{begin_pos}((ngày|đêm|hôm|sáng|trưa|tối|chiều|khuya|mùng|hôm qua|hôm nay)( +))*(\    
([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])( +)(tháng)( +)([1-9]|0[1-9]|1[0-2])( +)(năm)( +)([1-2][0-9][0-9][0-9])|\
([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])( +)(tháng)( +)([1-9]|0[1-9]|1[0-2])\
){end_pos}''',

    "mm_yyyy": f'''{begin_pos}(tháng)( +)(\
([1-9]|0[1-9]|1[0-2])( +)(năm)( +)([1-2][0-9][0-9][0-9])|\
([1-9]|0[1-9]|1[0-2])(\.|\-|\/)([1-2][0-9][0-9][0-9])|\
([1-9]|0[1-9]|1[0-2])\
){end_pos}''',

    "yyyy": f"{begin_pos}(năm)( +)([1-2][0-9][0-9][0-9]){end_pos}",    
    "raw_dd_mm_yyyy": f"{begin_pos}([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|\-|\/)([1-9]|0[1-9]|1[0-2])(\.|\-|\/)([1-2][0-9][0-9][0-9]){end_pos}",
    "raw_mm_yyyy": f"{begin_pos}([1-9]|0[1-9]|1[0-2])(\.|\-|\/)([1-2][0-9][0-9][0-9]){end_pos}",
    "raw_dd_mm": f"{begin_pos}([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|\-|\/)([1-9]|0[1-9]|1[0-2]){end_pos}",
}

date_patterrn_layer_2 = {
    "dd_mm_yyyy": f"{begin_pos}([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|\-|\/)([1-9]|0[1-9]|1[0-2])(\.|\-|\/)([1-2][0-9][0-9][0-9]){end_pos}",
    "mm_yyyy": f"{begin_pos}([1-9]|0[1-9]|1[0-2])(\.|\-|\/)([1-2][0-9][0-9][0-9]){end_pos}",
    "dd_mm": f"{begin_pos}([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|\-|\/)([1-9]|0[1-9]|1[0-2]){end_pos}",
    "yyyy": f"{begin_pos}([1-2][0-9][0-9][0-9]){end_pos}",
    "mm": f"{begin_pos}([1-9]|0[1-9]|1[0-2]){end_pos}",
    "dd": f"{begin_pos}([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1]){end_pos}",
}

time_pattern = {"normal_pattern": f"{begin_pos}([1-9]|0[1-9]|1[0-9]|2[0-4])(:|h)([0-5][0-9])*( +phút)*{end_pos}"}
number_pattern = {"normal_pattern": f"[0-9]+((\.|\,)[0-9]+)*"}
currency_pattern = {"normal_pattern": f"{begin_pos}[0-9]+({'|'.join(list(units_dict.keys()))}){end_pos}"}

def find_all(date_patterrn, text, return_first=False):
    match_list = []
    while text is not None and len(text) != 0:
        has_match = False
        for key, pattern in date_patterrn.items():
            match = re.search(pattern, text)
            if match:
                match_text = match.group(0)
                if return_first: 
                    match_text = re.sub('(\. | \.|(^ )+|( $)+|(^\.)+|(\.$)+)', '', match_text)
                    return match_text
                match_list.append(match_text)
                text = text.replace(text[match.start():match.end()], " ")
                has_match = True
                break
            else:
                continue
        
        if not has_match:
            break
    
    return match_list
        
def extract_str(type: RegType = None, text: str=""):
    if type == RegType.DATE:
        match_list = find_all(date_patterrn_layer_1, text)
        match_list = list(map(lambda text: find_all(date_patterrn_layer_2, text, return_first=True), match_list))
        return match_list
    
    elif type == RegType.TIME:
        match_list = find_all(time_pattern, text)
        match_list = list(map(lambda text: re.sub('(\. | \.|(^ )+|( $)+|(^\.)+|(\.$)+)', '', text), match_list))
        return match_list

    elif type == RegType.NUMBER:
        match_list = find_all(number_pattern, text)
        return match_list
    
    elif type == RegType.CURRENCY:
        match_list = find_all(currency_pattern, text)
        match_list = list(map(lambda text: re.sub('(\. | \.|(^ )+|( $)+|(^\.)+|(\.$)+)', '', text), match_list))
        return match_list
    
    return None
    
def replace_str(type: RegType = None, list_match: list=[], normalize_text: str=""):
    try:
        for text in list_match:
            return_string = ""
            if type == RegType.DATE:
                date_splited = re.split("[\-\.\/]", text)
                if len(date_splited) == 3:
                    dict_convert = ["", "tháng", "năm"]
                elif len(date_splited) == 2:
                    if len(date_splited[-1]) == 4:
                        dict_convert = ["", "năm"]
                    elif len(date_splited[-1]) <= 2:
                        dict_convert = ["", "tháng"]
                else:
                    return normalize_text
                
                for idx, value in enumerate(date_splited):
                    value = re.sub("^[0]+", "", value)
                    convert_text = n2w(value)
                    return_string += f"{dict_convert[idx]} {convert_text} "
                
            elif type == RegType.TIME:
                dict_convert = ["giờ", "phút"]
                text_splited = text.split()[0]
                text_splited = re.split("[\:h]", text_splited)
                for idx, value in enumerate(text_splited):
                    value = re.sub("^[0]+", "", value)
                    if value == "": continue
                    convert_text = n2w(value)
                    return_string += f"{convert_text} {dict_convert[idx]} "
            
            elif type == RegType.NUMBER:
                convert_text = re.sub('\.\,', '', text)
                return_string = n2w(convert_text)
            
            elif type == RegType.CURRENCY:
                convert_text = re.sub('\.\,', '', text)
                match_digit = re.search("[0-9]+", convert_text)
                match_unit = re.search(f"{'|'.join(list(units_dict.keys()))}", convert_text)
                if match_digit is None or match_unit is None: return normalize_text
                
                return_string = n2w(match_digit.group(0)) + " " + units_dict[f"\{match_unit.group(0)}" if match_unit.group(0) in ["$"] else match_unit.group(0)]
                # print(return_string)
                
            normalize_text = re.sub(text, f" {return_string} ", normalize_text)
                    
        return normalize_text
        
    except Exception as e:
        print(e)