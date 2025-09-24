import os
import json
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import random

# ==============================================================================
# BÖLÜM 1: KONTROL PANELİ 
# ==============================================================================

def get_config():
    """Proje için gerekli tüm yapılandırma ayarlarını yükler."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("HATA: OPENAI_API_KEY bulunamadı. Lütfen .env dosyası oluşturun.")

    system_prompt = """
    Sen 'BANÜ MIS CONNECT', Bandırma Onyedi Eylül Üniversitesi Yönetim Bilişim Sistemleri (YBS) bölümü öğrencileri için özel olarak tasarlanmış, 
    sektör lideri bir akıllı asistansın. Görevin, YBS öğrencilerinin akademik ve sosyal hayatını kolaylaştırmak; ders programları, akademik takvim, 
    kampüs etkinlikleri, kampüs içi mekanlar ve Bandırma hava durumu gibi bilgilere anında erişimlerini sağlamaktır.

    KİŞİLİK: Son derece profesyonel, proaktif, bilgili ve arkadaş canlısı bir tonda konuş. Cevapların daima net, anlaşılır ve eyleme geçirilebilir olsun.
    Araçlardan gelen JSON verisini asla ham olarak gösterme; bunun yerine veriyi yorumlayarak, öğrencinin anlayacağı şekilde, 
    akıcı cümlelerle sun. Bandırma ve BANÜ hakkında spesifik konuş, özellikle Uygulamalı Bilimler Fakültesi'ndeki YBS bölümünün ihtiyaçlarına odaklan.
    Önemli kelimeleri veya başlıkları vurgulamak için metni *kalın* olarak işaretle. Örneğin: *Ders Adı:* veya *Konum:*.
    """
    return {
        "api_key": api_key,
        "model_name": "gpt-4-turbo-preview",
        "system_prompt": system_prompt
    }

# ==============================================================================
# BÖLÜM 2: VERİ KAYNAĞI 
# ==============================================================================

COURSE_SCHEDULE = {
    "Pazartesi": [
        {"name": "Yönetim Bilişim Sistemlerine Giriş", "time": "10:00", "location": "Uygulamalı Bilimler Fak. (UBF) - Amfi 1", "instructor": "Doç. Dr. Alev ATEŞ"},
        {"name": "Bilgisayar Programlama I", "time": "14:00", "location": "UBF - Bilgisayar Lab. 1", "instructor": "Dr. Öğr. Üyesi Burcu YILMAZ"}
    ], "Salı": [
        {"name": "Sistem Analizi ve Tasarımı", "time": "09:00", "location": "UBF - D201", "instructor": "Prof. Dr. Cihan ÇETİNKAYA"},
        {"name": "İşletme Yönetimi", "time": "13:00", "location": "UBF - D105", "instructor": "Dr. Öğr. Üyesi Selin GÜNER"}
    ], "Çarşamba": [
        {"name": "Veritabanı Yönetim Sistemleri", "time": "11:00", "location": "UBF - Bilgisayar Lab. 2", "instructor": "Dr. Öğr. Üyesi Burcu YILMAZ"},
    ], "Perşembe": [
        {"name": "Sistem Analizi ve Tasarımı", "time": "09:00", "location": "UBF - D201", "instructor": "Prof. Dr. Cihan ÇETİNKAYA"},
        {"name": "Müşteri İlişkileri Yönetimi (CRM)", "time": "14:00", "location": "UBF - D303", "instructor": "Doç. Dr. Alev ATEŞ"}
    ], "Cuma": [
        {"name": "Bilgisayar Programlama I", "time": "10:00", "location": "UBF - Bilgisayar Lab. 1", "instructor": "Dr. Öğr. Üyesi Burcu YILMAZ"}
    ]
}

ACADEMIC_CALENDAR = {
    "ders_kayit_haftasi": {"start_date": "2025-02-10", "end_date": "2025-02-14", "description": "Bahar dönemi ders kayıtları ve danışman onayları haftası."},
    "vize_haftasi": {"start_date": "2025-04-15", "end_date": "2025-04-21", "description": "Bahar dönemi ara sınavları."},
    "final_haftasi": {"start_date": "2025-06-03", "end_date": "2025-06-14", "description": "Bahar dönemi final sınavları."},
    "bütünleme_haftasi": {"start_date": "2025-06-20", "end_date": "2025-06-27", "description": "Bahar dönemi bütünleme sınavları."},
    "ders_birakma_son_gunu": {"date": "2025-03-29", "description": "Ders bırakmak (withdrawal) için son tarih."},
    "bahar_tatili": {"start_date": "2025-04-08", "end_date": "2025-04-12", "description": "Bahar dönemi ara tatili."},
    "yatay_gecis_basvurulari": {"start_date": "2025-07-15", "end_date": "2025-07-25", "description": "Kurumlararası ve kurum içi yatay geçiş başvuru tarihleri."}
}

CAMPUS_EVENTS = [
    {"name": "YBS Zirvesi'25", "date": "2025-10-22T10:00:00", "location": "Prof. Dr. Fuat Sezgin Konferans Salonu", "category": "kariyer", "description": "YBS alanında önde gelen şirketlerin CEO ve yöneticilerinin katılacağı, sektörün geleceğinin konuşulacağı kariyer zirvesi."},
    {"name": "Bahar Şenliği Konseri: Duman", "date": "2025-05-21T20:30:00", "location": "Merkez Kampüs Stadyumu", "category": "konser", "description": "Geleneksel bahar şenlikleri kapsamında Duman sahne alacak. Giriş ücretsizdir."},
    {"name": "Python Atölyesi: Veri Analizine Giriş", "date": "2025-11-12T14:00:00", "location": "UBF - Bilgisayar Lab. 3", "category": "atölye", "description": "YBS Kulübü tarafından düzenlenen, veri analizine başlangıç seviyesinde Python eğitimi."},
    {"name": "E-Spor Turnuvası: League of Legends", "date": "2025-11-05T12:00:00", "location": "Öğrenci Yaşam Merkezi", "category": "turnuva", "description": "BANÜ E-Spor Kulübü tarafından düzenlenen ödüllü League of Legends turnuvası. Takım kayıtları devam ediyor."},
    {"name": "Kariyer Söyleşisi: CRM'in Geleceği", "date": "2025-10-29T15:00:00", "location": "UBF Konferans Salonu", "category": "seminer", "description": "Salesforce Türkiye direktörünün katılacağı, Müşteri İlişkileri Yönetimi'nin geleceği üzerine bir söyleşi."},
    {"name": "Fotoğrafçılık Kulübü Sergisi: Bandırma'nın Renkleri", "date": "2025-11-10T09:00:00", "location": "Rektörlük Binası Sergi Alanı", "category": "sergi", "description": "Fotoğrafçılık Kulübü üyelerinin Bandırma'da çektiği fotoğraflardan oluşan sergi."}
]

CAMPUS_LOCATIONS = {
    "uygulamalı bilimler fakültesi": {"description": "Merkez kampüs ana girişinin sağında, yemekhanenin karşısında yer alan yeni ve modern bina. YBS bölümü bu binanın 2. ve 3. katlarındadır.", "map_link": "https://maps.google.com/d/viewer?mid=12345"},
    "merkez kütüphane": {"description": "Merkez meydanın hemen yanında bulunan, 4 katlı modern cam bina.", "map_link": "https://maps.google.com/d/viewer?mid=12345"},
    "öğrenci işleri daire başkanlığı": {"description": "Rektörlük binasının zemin katında yer almaktadır. Öğrenci belgesi, transkript gibi işlemler buradan yapılır.", "map_link": "https://maps.google.com/d/viewer?mid=12345"},
    "mediko sosyal merkezi": {"description": "Yemekhanenin yanında bulunan, öğrencilere sağlık hizmeti sunan merkez.", "map_link": "https://maps.google.com/d/viewer?mid=12345"},
    "öğrenci yaşam merkezi": {"description": "İçerisinde çeşitli öğrenci kulüplerinin ofisleri, kafeteryalar ve sosyal alanlar bulunan merkez.", "map_link": "https://maps.google.com/d/viewer?mid=12345"}
}

YEMEKHANE_MENU = {
    "Pazartesi": "Mercimek Çorbası, Orman Kebabı, Pirinç Pilavı, Salata", "Salı": "Yayla Çorbası, Tavuk Döner, Patates Kızartması, Ayran",
    "Çarşamba": "Domates Çorbası, Izgara Köfte, Bulgur Pilavı, Mevsim Salata", "Perşembe": "Tavuk Suyu Çorba, Fırında Balık, Makarna, Salata",
    "Cuma": "Ezogelin Çorbası, Kuru Fasulye, Pirinç Pilavı, Turşu"
}

STUDENT_CLUBS = {
    "YBS Kulübü": "Yönetim Bilişim Sistemleri öğrencilerini sektörle buluşturan, teknik eğitimler ve kariyer etkinlikleri düzenleyen bir kulüptür.",
    "Girişimcilik Kulübü": "İş fikri olan öğrencilere mentorluk desteği sağlar ve yatırımcı buluşmaları organize eder.",
    "Veri Bilimi Kulübü": "Veri analizi, makine öğrenmesi gibi konularda atölyeler ve yarışmalar düzenler.",
    "E-Spor Kulübü": "Çeşitli online oyunlarda turnuvalar düzenler ve üniversite takımlarını yönetir."
}

# ==============================================================================
# BÖLÜM 3: TOOL BOX (ARAÇLAR VE YARDIMCI FONKSİYONLAR)
# ==============================================================================

def get_bandirma_weather() -> str:
    conditions = ["Güneşli", "Parçalı Bulutlu", "Yağmurlu", "Rüzgarlı", "Açık"]
    weather = { "city": "Bandırma", "temperature": random.randint(15, 25), "condition": random.choice(conditions), "humidity": random.randint(40, 70), "wind_speed": random.randint(5, 20) }
    return json.dumps(weather, ensure_ascii=False)

def get_clothing_suggestion(weather_condition: str, temperature: int) -> str:
    if "yağmur" in weather_condition.lower(): suggestion = "Hava yağmurlu görünüyor, yanına şemsiye veya yağmurluk almayı unutma!"
    elif temperature > 22: suggestion = "Bugün hava oldukça sıcak. Tişört gibi hafif ve rahat kıyafetler tercih edebilirsin."
    elif temperature < 18: suggestion = "Hava biraz serin. Üzerine bir ceket veya sweatshirt alman iyi olabilir."
    else: suggestion = "Hava ılık, mevsimlik kıyafetler için harika bir gün."
    return json.dumps({"suggestion": suggestion}, ensure_ascii=False)

def calculate_gano(grades_and_credits: str) -> str:
    """
    Anlık GANO Hesaplayıcı: Kullanıcıdan 'AA,3,BA,3,CC,4' gibi not ve kredi çiftleri alır,
    GANO'yu hesaplar ve sonucu JSON olarak döndürür.
    """
    grade_map = {"AA": 4.0, "BA": 3.5, "BB": 3.0, "CB": 2.5, "CC": 2.0, "DC": 1.5, "DD": 1.0, "FD": 0.5, "FF": 0.0}
    try:
        items = grades_and_credits.replace(" ", "").split(',')
        if len(items) % 2 != 0:
            return json.dumps({"error": "Lütfen her not için bir kredi girin. Örneğin: AA,3,BA,3"})
        total_credit_points, total_credits = 0, 0
        for i in range(0, len(items), 2):
            grade, credit = items[i].upper(), int(items[i+1])
            if grade not in grade_map:
                return json.dumps({"error": f"Geçersiz not: {grade}. Lütfen standart harf notlarını kullanın."})
            total_credit_points += grade_map[grade] * credit
            total_credits += credit
        if total_credits == 0:
            return json.dumps({"action": "show_gano_result", "data": {"gano": 0, "total_credits": 0}})
        gano = round(total_credit_points / total_credits, 2)
        return json.dumps({"action": "show_gano_result", "data": {"gano": gano, "total_credits": total_credits}})
    except Exception as e:
        return json.dumps({"error": f"Hesaplama hatası: Girdi formatınızı kontrol edin. Hata: {str(e)}"})

def create_study_plan(subject: str, hours: int) -> str:
    """
    Akıllı Pomodoro Planlayıcı: Kullanıcıdan ders adı ve toplam saat alır,
    Pomodoro tekniğine uygun çalışma/mola bloklarını JSON olarak döndürür.
    """
    if not isinstance(hours, int) or not (0 < hours < 10):
        return json.dumps({"error": "Lütfen 1 ile 9 saat arasında geçerli bir süre girin."})
    total_minutes, pomodoro_duration, short_break_duration, plan = hours * 60, 45, 10, []
    current_time = 0
    while current_time < total_minutes:
        work_end = min(current_time + pomodoro_duration, total_minutes)
        plan.append({"type": "work", "duration": work_end - current_time})
        current_time = work_end
        if current_time >= total_minutes: break
        break_end = min(current_time + short_break_duration, total_minutes)
        plan.append({"type": "break", "duration": break_end - current_time})
        current_time = break_end
    return json.dumps({"action": "show_pomodoro", "data": {"subject": subject, "plan": plan}}, ensure_ascii=False)

def get_yemekhane_menu(day: str = None) -> str:
    day_map = {0: 'Pazartesi', 1: 'Salı', 2: 'Çarşamba', 3: 'Perşembe', 4: 'Cuma'}
    target_day = day.capitalize() if day else day_map.get(datetime.now().weekday(), "Hafta sonu")
    menu = YEMEKHANE_MENU.get(target_day, "Bugün veya belirtilen günde yemekhane hizmet vermemektedir.")
    return json.dumps({target_day: menu}, ensure_ascii=False)
    
def get_student_clubs(query: str = None) -> str:
    if query:
        results = {name: desc for name, desc in STUDENT_CLUBS.items() if query.lower() in name.lower()}
        if results: return json.dumps(results, ensure_ascii=False)
        return json.dumps({"error": f"'{query}' ile ilgili bir kulüp bulunamadı."})
    return json.dumps(STUDENT_CLUBS, ensure_ascii=False)

def get_course_schedule(day: str = None) -> str:
    if day:
        normalized_day = day.capitalize()
        schedule_for_day = COURSE_SCHEDULE.get(normalized_day)
        if schedule_for_day: return json.dumps({normalized_day: schedule_for_day}, ensure_ascii=False)
        return json.dumps({"error": f"'{day}' için ders programı bulunamadı."}, ensure_ascii=False)
    return json.dumps(COURSE_SCHEDULE, ensure_ascii=False)

def get_academic_calendar(event_type: str = None) -> str:
    if event_type:
        results = {k: v for k, v in ACADEMIC_CALENDAR.items() if event_type.lower() in k.lower()}
        if results: return json.dumps({"action": "show_calendar_view", "data": results}, ensure_ascii=False)
        return json.dumps({"error": f"'{event_type}' ile ilgili bir takvim etkinliği bulunamadı."}, ensure_ascii=False)
    return json.dumps({"action": "show_calendar_view", "data": ACADEMIC_CALENDAR}, ensure_ascii=False)

def find_campus_location(building_name: str) -> str:
    query = building_name.lower().replace('i', 'ı').replace('ubf', 'uygulamalı bilimler fakültesi')
    for name, details in CAMPUS_LOCATIONS.items():
        if query in name.lower():
            return json.dumps({ "action": "show_location_card", "data": {"location": name.capitalize(), "details": details} }, ensure_ascii=False)
    return json.dumps({"error": f"'{building_name}' adında bir yer kampüs içinde bulunamadı."}, ensure_ascii=False)

def get_campus_events(category: str = None) -> str:
    now = datetime.now()
    upcoming_events = sorted([e for e in CAMPUS_EVENTS if datetime.fromisoformat(e['date']) > now], key=lambda x: x['date'])
    
    if category:
        filtered_events = [e for e in upcoming_events if category.lower() in e['category'].lower()]
        return json.dumps({"action": "show_event_cards", "data": filtered_events}, ensure_ascii=False)
    return json.dumps({"action": "show_event_cards", "data": upcoming_events}, ensure_ascii=False)

def get_tool_definitions() -> list:
    return [
        {"type": "function", "function": {"name": "get_bandirma_weather", "description": "Bandırma için anlık hava durumu bilgisini getirir.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "get_clothing_suggestion", "description": "Hava durumuna göre giyim tavsiyesi verir.", "parameters": {"type": "object", "properties": {"weather_condition": {"type": "string"}, "temperature": {"type": "number"}}, "required": ["weather_condition", "temperature"]}}},
        {"type": "function", "function": {"name": "calculate_gano", "description": "Öğrencinin GANO'sunu hesaplar ve sonucu yan panelde gösterir. Girdi formatı: 'Not1,Kredi1,Not2,Kredi2'.", "parameters": {"type": "object", "properties": {"grades_and_credits": {"type": "string", "description": "Örnek: 'AA,3,BA,3,CC,4'"}}, "required": ["grades_and_credits"]}}},
        {"type": "function", "function": {"name": "get_yemekhane_menu", "description": "BANÜ yemekhanesinin menüsünü getirir.", "parameters": {"type": "object", "properties": {"day": {"type": "string"}}, "required": []}}},
        {"type": "function", "function": {"name": "get_student_clubs", "description": "BANÜ'deki öğrenci kulüpleri hakkında bilgi verir.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": []}}},
        {"type": "function", "function": {"name": "get_course_schedule", "description": "YBS bölümü ders programını alır.", "parameters": {"type": "object", "properties": {"day": {"type": "string"}}, "required": []}}},
        {"type": "function", "function": {"name": "get_academic_calendar", "description": "BANÜ akademik takvimini sorgular ve yan panelde gösterir.", "parameters": {"type": "object", "properties": {"event_type": {"type": "string", "description": "Örn: 'vize', 'final', 'tatil'"}}, "required": []}}},
        {"type": "function", "function": {"name": "find_campus_location", "description": "BANÜ kampüsündeki bir mekanın yerini bulur ve yan panelde gösterir.", "parameters": {"type": "object", "properties": {"building_name": {"type": "string", "description": "Örn: 'UBF', 'kütüphane'"}}, "required": ["building_name"]}}},
        {"type": "function", "function": {"name": "get_campus_events", "description": "Yaklaşan kampüs etkinliklerini bulur ve yan panelde kartlar halinde gösterir.", "parameters": {"type": "object", "properties": {"category": {"type": "string", "description": "Etkinlik kategorisi, örn: 'konser', 'seminer'"}}, "required": []}}},
        {"type": "function", "function": {"name": "create_study_plan", "description": "Belirli bir ders için Pomodoro tekniğine dayalı, interaktif bir çalışma planı ve sayacı oluşturur.", "parameters": {"type": "object", "properties": {"subject": {"type": "string", "description": "Çalışılacak dersin adı"}, "hours": {"type": "number", "description": "Toplam çalışma süresi (saat olarak)"}}, "required": ["subject", "hours"]}}}
    ]

AVAILABLE_TOOLS = {
    "get_bandirma_weather": get_bandirma_weather,
    "get_clothing_suggestion": get_clothing_suggestion,
    "calculate_gano": calculate_gano,
    "get_yemekhane_menu": get_yemekhane_menu,
    "get_student_clubs": get_student_clubs,
    "get_course_schedule": get_course_schedule,
    "get_academic_calendar": get_academic_calendar,
    "find_campus_location": find_campus_location,
    "get_campus_events": get_campus_events,
    "create_study_plan": create_study_plan
}

# ==============================================================================
# BÖLÜM 4: MIS CONNECT CHATBOT (YBS'YE ÖZEL DİYALOG YÖNETİMİ)
# ==============================================================================

class CampusAssistant:
    def __init__(self):
        self.config = get_config()
        self.client = OpenAI(api_key=self.config['api_key'])
        self.tool_schemas = get_tool_definitions()
        self.available_tools = AVAILABLE_TOOLS
        self.reset_conversation()

    def reset_conversation(self):
        self.conversation_history = [{"role": "system", "content": self.config['system_prompt']}]

    def run_conversation(self, user_message: str, history: list = None) -> dict:
        if history:
            self.conversation_history = history
        
        self.conversation_history.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.config['model_name'], messages=self.conversation_history,
                tools=self.tool_schemas, tool_choice="auto",
            )
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                self.conversation_history.append(response_message)
                action_data = {}
                weather_data_for_suggestion = None

                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    if function_name == 'get_bandirma_weather':
                        weather_str = get_bandirma_weather()
                        weather_data_for_suggestion = json.loads(weather_str)
                        self.conversation_history.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": weather_str})
                        continue

                    function_to_call = self.available_tools.get(function_name)
                    if function_to_call:
                        function_args = json.loads(tool_call.function.arguments)
                        function_response_str = function_to_call(**function_args)
                        function_response_json = json.loads(function_response_str)
                        if "action" in function_response_json:
                            action_data = {"action": function_response_json["action"], "data": function_response_json["data"]}
                        self.conversation_history.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": function_response_str})
                
                if weather_data_for_suggestion:
                    suggestion_str = get_clothing_suggestion(weather_data_for_suggestion['condition'], weather_data_for_suggestion['temperature'])
                    self.conversation_history.append({
                        "role": "assistant", "content": None,
                        "tool_calls": [{"id": "call_clothing_suggestion", "type": "function", "function": {"name": "get_clothing_suggestion", "arguments": json.dumps(weather_data_for_suggestion)}}]
                    })
                    self.conversation_history.append({"tool_call_id": "call_clothing_suggestion", "role": "tool", "name": "get_clothing_suggestion", "content": suggestion_str})

                second_response = self.client.chat.completions.create(model=self.config['model_name'], messages=self.conversation_history)
                final_answer = second_response.choices[0].message.content
                self.conversation_history.append({"role": "assistant", "content": final_answer})
                
                response_payload = {"response": final_answer, "history": self.conversation_history}
                if action_data: response_payload.update(action_data)
                return response_payload
            else:
                final_answer = response_message.content
                self.conversation_history.append({"role": "assistant", "content": final_answer})
                return {"response": final_answer, "history": self.conversation_history}
        except Exception as e:
            print(f"HATA: API çağrısı sırasında bir sorun oluştu: {e}")
            return {"response": "Üzgünüm, bir teknik sorun yaşıyorum. Lütfen daha sonra tekrar deneyin."}

# ==============================================================================
# BÖLÜM 5: FLASK WEB UYGULAMASI 
# ==============================================================================

app = Flask(__name__)
assistant = CampusAssistant()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BANÜ MIS CONNECT | Akıllı YBS Asistanı</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #0d3b66; --secondary-color: #00a896; --accent-color: #f4d35e;
            --background-color-start: #141E30; --background-color-end: #243B55;
            --surface-color: rgba(255, 255, 255, 0.1); --text-color: #F0F4F8;
            --border-color: rgba(255, 255, 255, 0.2); --font-family: 'Inter', sans-serif;
            --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
        }
        * { box-sizing: border-box; }
        body { 
            font-family: var(--font-family); margin: 0;
            background: linear-gradient(-45deg, var(--background-color-start), var(--background-color-end), #0d3b66, #00a896);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color: var(--text-color); display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 1rem;
        }
        @keyframes gradientBG { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
        .app-container { width: 100%; max-width: 1400px; display: grid; grid-template-columns: 280px minmax(0, 2fr) minmax(320px, 1fr); gap: 1.5rem; height: 95vh; max-height: 950px;}
        .history-panel, .main-chat-panel, .side-panel-card { 
            background: var(--surface-color); border-radius: 24px; border: 1px solid var(--border-color);
            backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
            transition: all 0.3s ease; display: flex; flex-direction: column;
        }
        /* History Panel */
        .history-panel { padding: 1rem; }
        .history-header { padding: 0.5rem; text-align: center; font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; }
        #new-chat-btn { background-color: var(--secondary-color); color: white; border: none; width: 100%; padding: 0.75rem; border-radius: 12px; font-weight: 600; cursor: pointer; transition: all .2s ease; }
        #new-chat-btn:hover { background-color: var(--accent-color); color: var(--primary-color); }
        .history-list { margin-top: 1rem; flex-grow: 1; overflow-y: auto; }
        .history-item { padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem; cursor: pointer; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; transition: all .2s ease; display: flex; justify-content: space-between; align-items: center;}
        .history-item:hover, .history-item.active { background-color: rgba(255,255,255,0.2); }
        .delete-chat-btn { background: none; border: none; color: #ff5c5c; cursor: pointer; opacity: 0; transition: opacity .2s; }
        .history-item:hover .delete-chat-btn { opacity: 1; }

        /* Main Chat Panel */
        .main-chat-panel { box-shadow: var(--shadow-lg); }
        .chat-header { padding: 1rem 1.5rem; color: white; display: flex; align-items: center; gap: 1rem; border-bottom: 1px solid var(--border-color); }
        .chat-header-icon { font-size: 2.5rem; line-height: 1; filter: drop-shadow(0 0 5px var(--secondary-color));}
        .chat-header h1 { margin: 0; font-size: 1.25rem; font-weight: 600; }
        .chat-messages { flex-grow: 1; padding: 1.5rem; overflow-y: auto; display: flex; flex-direction: column; gap: 1rem; }
        .message { display: flex; max-width: 85%; align-items: flex-start; gap: 0.75rem; opacity: 0; animation: fadeIn 0.5s forwards; }
        .message.user { margin-left: auto; flex-direction: row-reverse; }
        .message-avatar { width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; color: var(--primary-color); flex-shrink: 0; background-color: var(--accent-color); }
        .message.bot .message-avatar { background-color: var(--secondary-color); color: white; }
        .message-bubble { padding: 0.75rem 1.25rem; border-radius: 1.25rem; line-height: 1.6; word-wrap: break-word; background: rgba(0,0,0,0.2); }
        .message.user .message-bubble { background: var(--secondary-color); color: white; border-bottom-right-radius: 0.25rem; }
        .message.bot .message-bubble { background: rgba(255, 255, 255, 0.15); color: var(--text-color); border-bottom-left-radius: 0.25rem; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .starters { padding: 1rem 1.5rem; border-top: 1px solid var(--border-color); }
        .starters-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
        .starter-chip { background-color: rgba(255, 255, 255, 0.1); border: 1px solid var(--border-color); color: var(--text-color); border-radius: 12px; padding: 0.75rem; font-size: 0.875rem; cursor: pointer; transition: all 0.2s ease; text-align: left; display: flex; align-items: center; gap: 0.5rem; }
        .starter-chip:hover { border-color: var(--secondary-color); background-color: rgba(0, 168, 150, 0.2); }
        .chat-input-area { padding: 1rem 1.5rem; border-top: 1px solid var(--border-color); display: flex; gap: 0.75rem; }
        #user-input { flex-grow: 1; background-color: rgba(0,0,0,0.2); border: 2px solid var(--border-color); border-radius: 999px; padding: 0.75rem 1.25rem; font-size: 1rem; transition: all 0.2s ease; color: white; }
        #user-input::placeholder { color: rgba(255,255,255,0.5); }
        #user-input:focus { border-color: var(--secondary-color); box-shadow: 0 0 10px var(--secondary-color); outline: none; }
        #send-button { background-color: var(--secondary-color); color: white; border: none; border-radius: 50%; width: 50px; height: 50px; cursor: pointer; transition: all 0.2s ease; display: flex; justify-content: center; align-items: center; flex-shrink: 0; }
        #send-button:hover { background-color: var(--accent-color); color: var(--primary-color); transform: scale(1.1) rotate(15deg); }
        
        /* Side Panel */
        .side-panel { display: flex; flex-direction: column; gap: 1.5rem; }
        .side-panel-card { padding: 1.5rem; display: none; opacity: 0; animation: fadeIn 0.5s forwards; }
        .card-header { font-size: 1.1rem; font-weight: 600; color: var(--accent-color); margin-bottom: 1rem; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border-color); display: flex; align-items: center; gap: 0.5rem; }
        #location-card-image { width: 100%; height: 150px; border-radius: 12px; margin-bottom: 1rem; text-align: center; line-height: 150px; font-size: 3rem; background-color: rgba(0,0,0,0.2); }
        #location-card-description { font-size: 0.95rem; line-height: 1.6; }
        .event-list, .calendar-list { list-style: none; padding: 0; margin: 0; }
        .event-item, .calendar-item { padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem; background: rgba(0,0,0,0.2); }
        #gano-result-display { font-size: 2.5rem; font-weight: 700; text-align: center; color: var(--accent-color); }
        #pomodoro-plan li { display: flex; align-items: center; padding: 0.75rem; border-radius: 8px; transition: background-color 0.3s; }
        #pomodoro-plan li.active { background-color: rgba(0, 168, 150, 0.2); }
        .pomodoro-timer { font-weight: 600; }

    </style>
</head>
<body>
    <div class="app-container">
        <div class="history-panel">
            <div class="history-header">Sohbet Geçmişi</div>
            <button id="new-chat-btn">Yeni Sohbet +</button>
            <div class="history-list" id="history-list"></div>
        </div>
        <div class="main-chat-panel">
            <header class="chat-header"><span class="chat-header-icon">🛰️</span><h1>BANÜ MIS CONNECT</h1></header>
            <div class="chat-messages" id="chat-messages"></div>
            <div class="starters" id="starters">
                <div class="starters-grid"></div>
            </div>
            <footer class="chat-input-area">
                <input type="text" id="user-input" placeholder="YBS Asistanına bir soru sor...">
                <button id="send-button" aria-label="Gönder">
                    <svg xmlns="http://www.w.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                </button>
            </footer>
        </div>
        <div class="side-panel" id="side-panel">
             <div class="side-panel-card" id="location-card">
                <div class="card-header" id="location-card-header"></div><div id="location-card-image"></div><div id="location-card-description"></div>
             </div>
             <div class="side-panel-card" id="events-card">
                <div class="card-header">📅 Yaklaşan Etkinlikler</div><ul class="event-list" id="event-list"></ul>
             </div>
             <div class="side-panel-card" id="calendar-card">
                <div class="card-header">🗓️ Akademik Takvim</div><ul class="calendar-list" id="calendar-list"></ul>
             </div>
              <div class="side-panel-card" id="gano-card">
                <div class="card-header">📊 GANO Sonucu</div>
                <div id="gano-result-display">0.00</div>
                <div id="gano-details" style="text-align: center; margin-top: 0.5rem;"></div>
             </div>
             <div class="side-panel-card" id="pomodoro-card">
                <div class="card-header" id="pomodoro-subject">Çalışma Planı</div>
                <ul id="pomodoro-plan" style="list-style: none; padding: 0;"></ul>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const chatMessages = document.getElementById('chat-messages');
            const userInput = document.getElementById('user-input');
            const sendButton = document.getElementById('send-button');
            const startersGrid = document.querySelector('.starters-grid');
            const newChatBtn = document.getElementById('new-chat-btn');
            const historyList = document.getElementById('history-list');

            let chatHistory = [];
            let conversations = {};
            let activeChatId = null;
            let pomodoroInterval;

            const STARTERS = [
                { text: "Yaklaşan etkinlikler neler?", icon: "🎉" },
                { text: "Bandırma'da hava nasıl?", icon: "🌤️" },
                { text: "Akademik takvimde ne var?", icon: "🗓️" },
                { text: "YBS Kulübü hakkında bilgi", icon: "🤝" }
            ];

            const getWelcomeMessage = () => { /* ... (no changes) ... */ return new Date().getHours() < 12 ? "Günaydın!" : new Date().getHours() < 18 ? "İyi günler!" : "İyi akşamlar!"; };
            const scrollToBottom = () => { chatMessages.scrollTop = chatMessages.scrollHeight; };

            const addMessage = (text, type) => {
                const messageEl = document.createElement('div');
                messageEl.className = `message ${type}`;
                const avatarChar = type === 'user' ? 'Ö' : 'B';
                const avatar = `<div class="message-avatar">${avatarChar}</div>`;
                const formattedText = text.replace(/\\*([^*]+)\\*/g, '<strong>$1</strong>');
                const bubble = `<div class="message-bubble">${formattedText}</div>`;
                messageEl.innerHTML = type === 'user' ? bubble + avatar : avatar + bubble;
                chatMessages.appendChild(messageEl);
                scrollToBottom();
            };

            const handleSendMessage = async (messageText) => {
                const trimmedMessage = messageText.trim();
                if (trimmedMessage === '') return;
                
                if (!activeChatId) {
                    startNewChat();
                }

                addMessage(trimmedMessage, 'user');
                chatHistory.push({ role: 'user', content: trimmedMessage });
                userInput.value = '';
                sendButton.disabled = true;
                document.getElementById('starters').style.display = 'none';
                showTypingIndicator();

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: trimmedMessage, history: chatHistory.slice(0, -1)})
                    });
                    if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                    const data = await response.json();
                    
                    hideTypingIndicator();
                    addMessage(data.response, 'bot');
                    
                    chatHistory = data.history;
                    saveConversations();

                    // YAN PANEL EYLEMLERİNİ İŞLE
                    hideAllSidePanelCards();
                    if (data.action === 'show_location_card') handleLocationCard(data.data);
                    if (data.action === 'show_event_cards') handleEventCards(data.data);
                    if (data.action === 'show_calendar_view') handleCalendarView(data.data);
                    if (data.action === 'show_gano_result') handleGanoResult(data.data);
                    if (data.action === 'show_pomodoro') handlePomodoro(data.data);

                } catch (error) {
                    console.error('Fetch error:', error);
                    hideTypingIndicator();
                    addMessage('Üzgünüm, bir bağlantı hatası oluştu.', 'bot');
                } finally {
                    sendButton.disabled = false;
                    userInput.focus();
                }
            };
            
            const hideAllSidePanelCards = () => {
                 document.querySelectorAll('.side-panel-card').forEach(c => c.style.display = 'none');
            };

            const handleLocationCard = (data) => {
                const card = document.getElementById('location-card');
                document.getElementById('location-card-header').innerHTML = `📍 ${data.location}`;
                document.getElementById('location-card-description').innerHTML = data.details.description;
                document.getElementById('location-card-image').innerHTML = '🗺️';
                card.style.display = 'block';
            };

            const handleEventCards = (data) => {
                const card = document.getElementById('events-card');
                const list = document.getElementById('event-list');
                list.innerHTML = '';
                if(data.length === 0) {
                    list.innerHTML = '<li>Yaklaşan bir etkinlik bulunamadı.</li>';
                } else {
                    data.forEach(event => {
                        const date = new Date(event.date);
                        const formattedDate = `${date.toLocaleDateString('tr-TR')} ${date.toLocaleTimeString('tr-TR', {hour: '2-digit', minute:'2-digit'})}`;
                        list.innerHTML += `<li class="event-item"><strong>${event.name}</strong><br><small>${formattedDate} - ${event.location}</small></li>`;
                    });
                }
                card.style.display = 'block';
            };

            const handleCalendarView = (data) => {
                const card = document.getElementById('calendar-card');
                const list = document.getElementById('calendar-list');
                list.innerHTML = '';
                for (const key in data) {
                    const event = data[key];
                    const date = event.date ? new Date(event.date).toLocaleDateString('tr-TR') : `${new Date(event.start_date).toLocaleDateString('tr-TR')} - ${new Date(event.end_date).toLocaleDateString('tr-TR')}`;
                    list.innerHTML += `<li class="calendar-item"><strong>${event.description}</strong><br><small>${date}</small></li>`;
                }
                card.style.display = 'block';
            };

             const handleGanoResult = (data) => {
                const card = document.getElementById('gano-card');
                document.getElementById('gano-result-display').textContent = data.gano.toFixed(2);
                document.getElementById('gano-details').textContent = `${data.total_credits} kredi üzerinden hesaplandı.`;
                card.style.display = 'block';
            };

            const handlePomodoro = (data) => {
                const card = document.getElementById('pomodoro-card');
                document.getElementById('pomodoro-subject').textContent = `${data.subject} | Çalışma Planı`;
                const planList = document.getElementById('pomodoro-plan');
                planList.innerHTML = '';
                
                data.plan.forEach(block => {
                    const li = document.createElement('li');
                    const icon = block.type === 'work' ? '📚' : '☕️';
                    li.innerHTML = `<span>${icon}</span>&nbsp;
                                    <div>
                                        <span>${block.type === 'work' ? 'Çalışma' : 'Mola'}</span>
                                        <div class="pomodoro-timer" data-duration="${block.duration * 60}">
                                            ${block.duration}:00
                                        </div>
                                    </div>`;
                    planList.appendChild(li);
                });

                card.style.display = 'block';
                let totalSecondsPassed = 0;
                clearInterval(pomodoroInterval);
                pomodoroInterval = setInterval(() => {
                    totalSecondsPassed++;
                    let cumulativeSeconds = 0;
                    let finished = true;
                    document.querySelectorAll('#pomodoro-plan li').forEach((li) => {
                        const timerEl = li.querySelector('.pomodoro-timer');
                        const blockDuration = parseInt(timerEl.dataset.duration, 10);
                        cumulativeSeconds += blockDuration;
                        
                        li.classList.remove('active');

                        if(totalSecondsPassed < cumulativeSeconds) {
                            if(finished) li.classList.add('active'); // Activate first unfinished block
                            const secondsLeftInBlock = cumulativeSeconds - totalSecondsPassed;
                            const m = Math.floor(secondsLeftInBlock / 60).toString().padStart(2, '0');
                            const s = Math.floor(secondsLeftInBlock % 60).toString().padStart(2, '0');
                            timerEl.textContent = `${m}:${s}`;
                            finished = false;
                        } else {
                            timerEl.textContent = '00:00';
                        }
                    });

                    if(finished) clearInterval(pomodoroInterval);
                }, 1000);
            };
            
            const showTypingIndicator = () => { /* ... (no changes) ... */ };
            const hideTypingIndicator = () => { /* ... (no changes) ... */ };

            // --- LOCALSTORAGE & HISTORY MANAGEMENT ---
            const saveConversations = () => {
                if (activeChatId) {
                    conversations[activeChatId] = chatHistory;
                    localStorage.setItem('banu-mis-connect-chats', JSON.stringify(conversations));
                    renderHistory();
                }
            };

            const loadConversations = () => {
                const saved = localStorage.getItem('banu-mis-connect-chats');
                conversations = saved ? JSON.parse(saved) : {};
                renderHistory();
            };

            const renderHistory = () => {
                historyList.innerHTML = '';
                Object.keys(conversations).forEach(id => {
                    const history = conversations[id];
                    if (history.length > 1) { // has at least one user message
                        const firstUserMsg = history.find(m => m.role === 'user').content;
                        const item = document.createElement('div');
                        item.className = 'history-item';
                        item.dataset.id = id;
                        if(id === activeChatId) item.classList.add('active');

                        const text = document.createElement('span');
                        text.textContent = firstUserMsg;
                        item.appendChild(text);

                        const delBtn = document.createElement('button');
                        delBtn.className = 'delete-chat-btn';
                        delBtn.innerHTML = '🗑️';
                        delBtn.onclick = (e) => {
                            e.stopPropagation();
                            delete conversations[id];
                            saveConversations();
                            if(activeChatId === id) startNewChat();
                        };
                        item.appendChild(delBtn);

                        item.onclick = () => loadChat(id);
                        historyList.appendChild(item);
                    }
                });
            };
            
            const loadChat = (id) => {
                activeChatId = id;
                chatHistory = conversations[id];
                chatMessages.innerHTML = '';
                chatHistory.forEach(msg => {
                    if (msg.role === 'user') addMessage(msg.content, 'user');
                    if (msg.role === 'assistant') addMessage(msg.content, 'bot');
                });
                document.getElementById('starters').style.display = 'none';
                renderHistory();
            };

            const startNewChat = () => {
                const newId = new Date().getTime().toString();
                activeChatId = newId;
                chatHistory = []; // Will be populated with system prompt on server
                chatMessages.innerHTML = '';
                addMessage(`${getWelcomeMessage()} Ben BANÜ MIS Connect, senin kişisel YBS Asistanın. Sana nasıl yardımcı olabilirim?`, 'bot');
                document.getElementById('starters').style.display = 'block';
                hideAllSidePanelCards();
                renderHistory();
            };
            
            STARTERS.forEach(starter => {
                const chip = document.createElement('button');
                chip.className = 'starter-chip';
                chip.innerHTML = `<span>${starter.icon}</span> <span>${starter.text}</span>`;
                chip.onclick = () => handleSendMessage(starter.text);
                startersGrid.appendChild(chip);
            });

            sendButton.addEventListener('click', () => handleSendMessage(userInput.value));
            userInput.addEventListener('keydown', e => e.key === 'Enter' && (e.preventDefault(), handleSendMessage(userInput.value)));
            newChatBtn.addEventListener('click', startNewChat);

            loadConversations();
            if (Object.keys(conversations).length === 0) {
                startNewChat();
            } else {
                loadChat(Object.keys(conversations)[0]); // Load the first chat on start
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    history = data.get('history', None)
    
    if not user_message:
        return jsonify({'error': 'Mesaj boş olamaz'}), 400
    
    # Eğer yeni bir sohbetse, sunucu tarafında geçmişi sıfırla
    if not history:
        assistant.reset_conversation()
    
    response_data = assistant.run_conversation(user_message, history)

    # ChatCompletionMessage nesneleri JSON'a çevrilemez, bu yüzden sadece temel Python dict/list/string döndürülmeli.
    # Özellikle 'history' listesindeki her mesajı dict'e dönüştür.
    def serialize_message(msg):
        # Eğer msg zaten dict ise, döndür
        if isinstance(msg, dict):
            # Bazı OpenAI nesneleri (ör: tool_call_id) olabilir, onları da düzelt
            return {k: serialize_message(v) for k, v in msg.items()}
        # Eğer msg OpenAI ChatCompletionMessage ise, dict'e çevir
        if hasattr(msg, 'role') and hasattr(msg, 'content'):
            return {
                "role": getattr(msg, 'role', None),
                "content": getattr(msg, 'content', None)
            }
        # Liste ise, her elemanı serialize et
        if isinstance(msg, list):
            return [serialize_message(m) for m in msg]
        # Diğer tipler için olduğu gibi döndür
        return msg

    if "history" in response_data:
        response_data["history"] = serialize_message(response_data["history"])

    return jsonify(response_data)

if __name__ == '__main__':
    try:
        get_config()
        print("="*60)
        print("✅ BANÜ MIS CONNECT | Akıllı YBS Asistanı başlatılıyor...")
        print("🌐 Uygulamaya erişmek için: http://127.0.0.1:5000")
        print("🔴 Sunucuyu durdurmak için terminalde CTRL+C tuşlarına basın.")
        print("="*60)
        app.run(debug=True, port=5000)
    except ValueError as e:
        print("="*60)
        print(f"❌ UYGULAMA BAŞLATILAMADI: {e}")
        print("Lütfen .env dosyası oluşturup OPENAI_API_KEY='sk-...' anahtarınızı ekleyin.")
        print("="*60)
        print("✅ BANÜ MIS CONNECT | Akıllı YBS Asistanı başlatılıyor...")
        print("🌐 Uygulamaya erişmek için: http://127.0.0.1:5000")
        print("🔴 Sunucuyu durdurmak için terminalde CTRL+C tuşlarına basın.")
        print("="*60)
        app.run(debug=True, port=5000)
    except ValueError as e:
        print("="*60)
        print(f"❌ UYGULAMA BAŞLATILAMADI: {e}")
        print("Lütfen .env dosyası oluşturup OPENAI_API_KEY='sk-...' anahtarınızı ekleyin.")
        print("="*60)

