"""Synthetic ADVERSARIAL PAIRS for the benchmark.

Each pair contains a SCAM message and a closely-themed LEGITIMATE message that
differ in CRITICAL INTENT (who is asked to reveal a secret, who requests money,
which channel is used) rather than a single swapped word. The scam and legitimate
members of a pair share an ``adversarial_pair_id`` so tests can assert integrity.

Structure per pair: ``{category, en:{scam,legit}, roman:{scam,legit}, urdu:{scam,legit}}``
"""
from __future__ import annotations

ADVERSARIAL_PAIRS: list[dict] = [
    {
        "category": "otp_direction",
        "signals_scam": ["otp_request", "urgency"],
        "signals_legit": ["reassuring:do_not_share_code"],
        "en": {
            "scam": "Send me the OTP immediately or your account will be blocked, I am from the security team.",
            "legit": "Your OTP is {code}. Never share this code with anyone, our staff will never ask for it.",
        },
        "roman": {
            "scam": "Main security team se hun, apna OTP foran bhejein warna account block ho jayega.",
            "legit": "Aap ka OTP {code} hai. Is code ko kisi se share na karein, hamara staff kabhi nahi mangta.",
        },
        "urdu": {
            "scam": "میں سیکیورٹی ٹیم سے ہوں، اپنا او ٹی پی فوراً بھیجیں ورنہ اکاؤنٹ بلاک ہو جائے گا۔",
            "legit": "آپ کا او ٹی پی {code} ہے۔ اس کوڈ کو کسی سے شیئر نہ کریں، ہمارا عملہ کبھی نہیں مانگتا۔",
        },
    },
    {
        "category": "google_notice",
        "signals_scam": ["impersonation_claim", "urgency", "credential_request"],
        "signals_legit": ["reassuring:no_harmful_request_detected"],
        "en": {
            "scam": "This is Google Support. Your account will be deleted in 24 hours, call us now and give your password to save it.",
            "legit": "Google security alert: a new sign-in was detected. If this was not you, review your settings. We never ask for your password.",
        },
        "roman": {
            "scam": "Yeh Google Support hai. Aap ka account 24 ghantay mein delete ho jayega, abhi call karein aur password dein.",
            "legit": "Google security alert: ek naya sign-in detect hua. Agar aap ne nahi kiya to settings dekhein. Hum password kabhi nahi mangte.",
        },
        "urdu": {
            "scam": "یہ گوگل سپورٹ ہے۔ آپ کا اکاؤنٹ چوبیس گھنٹوں میں حذف ہو جائے گا، ابھی کال کریں اور پاس ورڈ دیں۔",
            "legit": "گوگل سیکیورٹی الرٹ: ایک نیا سائن ان آیا۔ اگر آپ نے نہیں کیا تو سیٹنگز دیکھیں۔ ہم پاس ورڈ کبھی نہیں مانگتے۔",
        },
    },
    {
        "category": "job_offer",
        "signals_scam": ["payment_request", "urgency"],
        "signals_legit": ["reassuring:no_harmful_request_detected"],
        "en": {
            "scam": "You are hired! Earn {amount} daily from home. Pay a {amount2} registration fee today to activate your job account.",
            "legit": "We reviewed your CV and invite you to an interview at our office. There is no fee at any stage of our hiring process.",
        },
        "roman": {
            "scam": "Aap hire ho gaye! Ghar se rozana {amount} kamayein. Job account activate karne ke liye aaj {amount2} registration fee dein.",
            "legit": "Hum ne aap ka CV dekha aur office mein interview ke liye bulaya hai. Hamari hiring ke kisi marhalay par koi fee nahi.",
        },
        "urdu": {
            "scam": "آپ بھرتی ہو گئے! گھر سے روزانہ {amount} کمائیں۔ جاب اکاؤنٹ فعال کرنے کے لیے آج {amount2} رجسٹریشن فیس دیں۔",
            "legit": "ہم نے آپ کا سی وی دیکھا اور دفتر میں انٹرویو کے لیے بلایا ہے۔ ہماری بھرتی کے کسی مرحلے پر کوئی فیس نہیں۔",
        },
    },
    {
        "category": "transaction_alert",
        "signals_scam": ["credential_request", "otp_request", "urgency"],
        "signals_legit": ["reassuring:no_harmful_request_detected"],
        "en": {
            "scam": "Your transaction was blocked for security reasons. To release the funds now, share your PIN and the OTP you received.",
            "legit": "A transaction of {amount} was debited from your account. If you did not make it, call the number printed on your card.",
        },
        "roman": {
            "scam": "Aap ki transaction security wajah se block ho gayi. Funds abhi release karane ke liye apna PIN aur aaya hua OTP share karein.",
            "legit": "Aap ke account se {amount} debit hue. Agar aap ne nahi kiye to card par chapay number par call karein.",
        },
        "urdu": {
            "scam": "آپ کی ٹرانزیکشن سیکیورٹی وجہ سے بلاک ہو گئی۔ رقم ابھی جاری کرانے کے لیے اپنا پن اور موصول او ٹی پی شیئر کریں۔",
            "legit": "آپ کے اکاؤنٹ سے {amount} ڈبٹ ہوئے۔ اگر آپ نے نہیں کیے تو کارڈ پر چھپے نمبر پر کال کریں۔",
        },
    },
    {
        "category": "delivery_fee",
        "signals_scam": ["payment_request", "urgency", "suspicious_link_or_login"],
        "signals_legit": ["reassuring:no_harmful_request_detected"],
        "en": {
            "scam": "Your package is held at customs. Pay a {amount} clearance fee at {url} within 2 hours or it will be destroyed.",
            "legit": "Your parcel cleared customs with no charges and is out for delivery today. No payment is required to receive it.",
        },
        "roman": {
            "scam": "Aap ka package customs mein hai. 2 ghantay ke andar {url} par {amount} clearance fee dein warna tabah kar diya jayega.",
            "legit": "Aap ka parcel baghair charges ke customs se clear ho gaya aur aaj delivery par hai. Lene ke liye koi payment nahi chahiye.",
        },
        "urdu": {
            "scam": "آپ کا پیکج کسٹم میں ہے۔ دو گھنٹوں کے اندر {url} پر {amount} کلیئرنس فیس دیں ورنہ ضائع کر دیا جائے گا۔",
            "legit": "آپ کا پارسل بغیر چارجز کے کسٹم سے کلیئر ہو گیا اور آج ڈیلیوری پر ہے۔ لینے کے لیے کوئی ادائیگی نہیں چاہیے۔",
        },
    },
    {
        "category": "support_call",
        "signals_scam": ["impersonation_claim", "credential_request"],
        "signals_legit": ["reassuring:no_harmful_request_detected"],
        "en": {
            "scam": "Your computer is infected. Install this remote-access tool and give us control so we can fix it and take your login.",
            "legit": "Thanks for contacting support. Your ticket is {code}; our agent will never ask for your password or ask you to install remote tools.",
        },
        "roman": {
            "scam": "Aap ke computer mein virus hai. Yeh remote-access tool install karein aur hamein control dein taake theek karein aur login le lein.",
            "legit": "Support se rabta ka shukriya. Aap ka ticket {code} hai; hamara agent kabhi password nahi mangta na remote tool install karne ko kehta hai.",
        },
        "urdu": {
            "scam": "آپ کے کمپیوٹر میں وائرس ہے۔ یہ ریموٹ رسائی ٹول انسٹال کریں اور ہمیں کنٹرول دیں تاکہ ٹھیک کریں اور لاگ ان لے لیں۔",
            "legit": "سپورٹ سے رابطے کا شکریہ۔ آپ کا ٹکٹ {code} ہے؛ ہمارا ایجنٹ کبھی پاس ورڈ نہیں مانگتا نہ ریموٹ ٹول انسٹال کرنے کو کہتا ہے۔",
        },
    },
    {
        "category": "investment_pitch",
        "signals_scam": ["payment_request", "urgency"],
        "signals_legit": ["reassuring:no_harmful_request_detected"],
        "en": {
            "scam": "Guaranteed {amount} weekly profit with zero risk. Wire your money to my personal wallet before the group closes tonight.",
            "legit": "Here is the fund's official prospectus and past performance. All investments carry risk; decide after reading, no rush.",
        },
        "roman": {
            "scam": "Baghair risk ke har hafte guaranteed {amount} munafa. Aaj raat group band hone se pehle mere personal wallet mein paisay wire karein.",
            "legit": "Yeh fund ka official prospectus aur purani karkardagi hai. Har investment mein risk hota hai, parh kar sochein, koi jaldi nahi.",
        },
        "urdu": {
            "scam": "بغیر خطرہ کے ہر ہفتے ضمانت شدہ {amount} منافع۔ آج رات گروپ بند ہونے سے پہلے میرے ذاتی والٹ میں رقم بھیجیں۔",
            "legit": "یہ فنڈ کا آفیشل پراسپیکٹس اور ماضی کی کارکردگی ہے۔ ہر سرمایہ کاری میں خطرہ ہوتا ہے، پڑھ کر سوچیں، کوئی جلدی نہیں۔",
        },
    },
    {
        "category": "marketplace_qr",
        "signals_scam": ["payment_request", "urgency"],
        "signals_legit": ["reassuring:no_harmful_request_detected"],
        "en": {
            "scam": "I am the buyer, scan this QR code and enter your card PIN to receive my payment instantly.",
            "legit": "I am interested in your item. I will inspect it in person and pay cash on pickup at a public place.",
        },
        "roman": {
            "scam": "Main buyer hun, yeh QR code scan karein aur payment foran lene ke liye apna card PIN darj karein.",
            "legit": "Mujhe aap ka saman chahiye. Main khud check karoon ga aur public jagah pickup par cash doon ga.",
        },
        "urdu": {
            "scam": "میں خریدار ہوں، یہ کیو آر کوڈ اسکین کریں اور ادائیگی فوراً لینے کے لیے اپنا کارڈ پن درج کریں۔",
            "legit": "مجھے آپ کا سامان چاہیے۔ میں خود چیک کروں گا اور عوامی جگہ پک اپ پر نقد دوں گا۔",
        },
    },
    {
        "category": "account_security",
        "signals_scam": ["credential_request", "urgency"],
        "signals_legit": ["reassuring:no_harmful_request_detected"],
        "en": {
            "scam": "Your account will be permanently disabled today. Reply with your password and the {code} code to keep it active.",
            "legit": "Two-step verification is now on for your account. If this was not you, use the recovery options inside the official app.",
        },
        "roman": {
            "scam": "Aap ka account aaj hamesha ke liye disable ho jayega. Chalu rakhne ke liye password aur {code} code reply karein.",
            "legit": "Aap ke account par two-step verification on ho gayi. Agar aap ne nahi ki to official app ke recovery options istemal karein.",
        },
        "urdu": {
            "scam": "آپ کا اکاؤنٹ آج ہمیشہ کے لیے غیر فعال ہو جائے گا۔ چالو رکھنے کے لیے پاس ورڈ اور {code} کوڈ بھیجیں۔",
            "legit": "آپ کے اکاؤنٹ پر ٹو سٹیپ ویریفکیشن آن ہو گئی۔ اگر آپ نے نہیں کی تو آفیشل ایپ کے ریکوری آپشنز استعمال کریں۔",
        },
    },
    {
        "category": "prize_claim",
        "signals_scam": ["payment_request", "urgency"],
        "signals_legit": ["reassuring:no_harmful_request_detected"],
        "en": {
            "scam": "You won {amount} in our lucky draw. Pay a {amount2} tax within {deadline} to release your prize money.",
            "legit": "You earned {amount} in loyalty points. Redeem them any time in the app for free; no payment is ever needed.",
        },
        "roman": {
            "scam": "Aap ne lucky draw mein {amount} jeete. Prize money lene ke liye {deadline} ke andar {amount2} tax dein.",
            "legit": "Aap ne loyalty points mein {amount} kamaye. App mein kabhi bhi free redeem karein, koi payment nahi chahiye.",
        },
        "urdu": {
            "scam": "آپ نے لکی ڈرا میں {amount} جیتے۔ انعام کی رقم لینے کے لیے {deadline} کے اندر {amount2} ٹیکس دیں۔",
            "legit": "آپ نے وفاداری پوائنٹس میں {amount} کمائے۔ ایپ میں کبھی بھی مفت ریڈیم کریں، کوئی ادائیگی نہیں چاہیے۔",
        },
    },
]
