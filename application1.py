import streamlit as st
import base64
import os
import io
from PIL import Image

# ---------------------- Helper Functions ----------------------

def set_overlay_bg_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            data = f.read()
        encoded_img = base64.b64encode(data).decode()
        return f"data:image/png;base64,{encoded_img}"
    else:
        st.error(f"Image not found at: {image_path}")
        return ""
    
def encode_image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  
    buffer.seek(0)  
    return base64.b64encode(buffer.read()).decode("utf-8")

def inject_css():
    footer = """
    <style>
    footer {
        background-color: #0A043C;
        color: white;
        padding: 5px 0;;
        width: 100%;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: Arial, sans-serif;
        margin-top: 50px; 
    }

    .footer-left {
        margin-left: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
        color: white; 
    }

    .footer-right {
        margin-right: 20px;
        display: flex;
        gap: 15px;
    }

    .footer-icon img {
        width: 20px;
        height: 20px;
        color: white;
    }

    .footer-left span {
        color: white; 
    }
    </style>

    <footer>
        <div class="footer-left">
            <span>🕻 +961 03 456 789 </span>
            <span>✉ chatcourse@gmail.com</span>
        </div>
        <div class="footer-right">
            <a href="https://facebook.com" target="_blank" class="footer-icon">
                <img src="https://img.icons8.com/ios-glyphs/30/ffffff/facebook.png" alt="Facebook"/>
        </a>
            <a href="https://twitter.com" target="_blank" class="footer-icon">
                <img src="https://img.icons8.com/ios-glyphs/30/ffffff/twitter.png" alt="Twitter"/>
            </a>
            <a href="https://linkedin.com" target="_blank" class="footer-icon">
                <img src="https://img.icons8.com/ios-filled/50/ffffff/linkedin.png"/>
            </a>
            <a href="https://instagram.com" target="_blank" class="footer-icon">
                <img src="https://img.icons8.com/ios-filled/50/ffffff/instagram-new.png"/>
            </a>
            <a href="https://youtube.com" target="_blank" class="footer-icon">
                <img src="https://img.icons8.com/ios-filled/50/ffffff/youtube-play.png"/>
            </a>
        </div>
    </footer>
    """
    st.markdown(footer, unsafe_allow_html=True)
    image_path = r"img\online-course.png"
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode()

    st.markdown(
    f"""
    <div class="header-container">
        <!-- Logo and title -->
        <div class="logo-container">
            <img src="data:image/png;base64,{base64_image}" alt="ChatCourse Logo" style="width: 40px; height: 40px; vertical-align: middle;">
            <a href="#" class="logo">ChatCourse</a>
        </div>
    </div>
    <style>
        .header-container {{
            display: flex;
            align-items: center;
            background-color: white;
            padding: 10px 20px;
            border-bottom: 1px solid #ddd;
        }}
        .logo-container {{
            display: flex;
            align-items: center;
            gap: 10px; 
        }}
        .logo {{
            font-size: 28px;
            font-weight: bold;
            color:#0A043C;
            text-decoration: none;
        }}
    </style>
    """,
    unsafe_allow_html=True,
    )
    st.markdown(
        """  

        <style>
        .header-container {
            display: flex;
            justify-content: flex-start; 
            align-items: center;
            background-color: white;
            padding: 10px 20px;
            border-bottom: 1px solid #ddd;
        }
        .logo-container {
            display: flex;
            align-items: center; 
            gap: 10px; 
        }
        .logo {
            font-size: 28px;
            font-weight: bold;
            color: #0A043C;
            text-decoration: none;
        }
        .menu-container {
            display: flex;
            justify-content: center;
            gap: 10px;
            padding: 10px 0; 
            border-bottom: 1px solid #ddd;
            background-color: #f9f9f9;
        }
        .menu-container button {
            background-color: white;
            color: black;
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 8px 15px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.2s ease;
        }
        .menu-container button:hover {
            background-color: #f0f0f0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def inject_css2():
    torn_edge_path = r"img\overlay-top.png"
    torn_edge_url = set_overlay_bg_image(torn_edge_path)
    # Styling for the footer
    st.markdown("""
        <style>
            .overlay-container {{
                position: relative;
                width: 100%; 
                height: 300px; 
                margin: 0 auto; 
                background-image: url("{bg_image_url}");
                background-size: cover;
                background-position: top;
                display: flex;
                justify-content: center;
                align-items: center;
                overflow: hidden; 
            }}
            .footer {
                background-color: #1C1C44;
                color: white;
                padding: 50px;
                text-align: left;
            }
            .footer h4 {
                margin-bottom: 20px;
            }
            .footer p {
                margin: 0 0 10px;
            }
            .footer a {
                color: #A9A9A9;
                text-decoration: none;
            }
            .footer a:hover {
                color: #FFFFFF;
            }
            .social-icons a {
                margin-right: 10px;
                font-size: 20px;
                color: #A9A9A9;
            }
            .social-icons a:hover {
                color: #FFFFFF;
            }
        </style>
    """, unsafe_allow_html=True)

    # Footer layout
    st.markdown("""
    <div class="footer">
        <div>
            <h4>ChatCourse</h4>
            <p>ChatCourse is an intuitive platform for exploring and accessing online courses. It offers a user-friendly design, easy navigation, and personalized learning, making education accessible from anywhere.</p>
        </div>
        <div style="margin-top: 30px;">
            <h4>Get In Touch</h4>
            <p>📍 AUB,Beirut,Lebanon</p>
            <p>📞 +961 03 456 789</p>
            <p>✉️ chatcourse@gmail.com</p>
        </div>
    <div class="social-icons">
                    <a href="https://facebook.com" target="_blank" class="footer-icon">
                        <img src="https://img.icons8.com/ios-glyphs/30/ffffff/facebook.png" alt="Facebook"/>
                </a>
                        <a href="https://twitter.com" target="_blank" class="footer-icon">
                    <img src="https://img.icons8.com/ios-glyphs/30/ffffff/twitter.png" alt="Twitter"/>
                </a>
                <a href="https://linkedin.com" target="_blank" class="footer-icon">
                    <img src="https://img.icons8.com/ios-filled/50/ffffff/linkedin.png"/>
                </a>
                <a href="https://instagram.com" target="_blank" class="footer-icon">
                    <img src="https://img.icons8.com/ios-filled/50/ffffff/instagram-new.png"/>
                </a>
                <a href="https://youtube.com" target="_blank" class="footer-icon">
                    <img src="https://img.icons8.com/ios-filled/50/ffffff/youtube-play.png"/>
                </a>
    <div class="torn-edge"></div>

    """, unsafe_allow_html=True)

# ---------------------- Page Functions ----------------------
def contact_page():
    inject_css()
    
    bg_image_path = r"img\page-header.jpg"
    torn_edge_path = r"img\overlay-top.png"

    bg_image_url =set_overlay_bg_image(bg_image_path)
    torn_edge_url = set_overlay_bg_image(torn_edge_path)

    st.markdown(f"""
        <style>
            .overlay-container {{
                position: relative;
                width: 100%; 
                height: 300px; 
                margin: 0 auto; 
                background-image: url("{bg_image_url}");
                background-size: cover;
                background-position: center;
                display: flex;
                justify-content: center;
                align-items: center;
                overflow: hidden;
            }}
            .overlay {{
                background-color: rgba(58, 118, 240, 0.6); 
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 1;
            }}
            .overlay-content {{
                position: relative;
                z-index: 2;
                color: white;
                text-align: center;
                font-family: Arial, sans-serif;
            }}
            .overlay-content h1 {{
                font-size: 36px;
                font-weight: bold;
                margin: 0;
            }}
            .overlay-content h2 {{
                font-size: 18px;
                margin: 10px 0 0;
            }}
            .torn-edge {{
                position: absolute;
                bottom: -17px; 
                left: 0;
                width: 100%;
                height: 60px; 
                background-image: url("{torn_edge_url}");
                background-size: 100% 100%; 
                background-repeat: no-repeat;
                transform: scaleY(-1);
                z-index: 2; 
                margin-top: 0px;
            }}
        </style>
        <div class="overlay-container">
            <div class="overlay"></div>
            <div class="overlay-content">
                <h1>Contact</h1>
                <h2>Home &raquo; Contact</h2>
            </div>
        </div>
        <div class="torn-edge"></div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <style>
            .contact-left {
                background-color: #f4f8fd;
                padding: 20px;
                border-radius: 10px;
                width: 100%;
                margin-top: 50px; 
            }
            .contact-item {
                display: flex;
                align-items: center;
                margin-bottom: 20px;
            }
            .contact-item-icon {
                width: 50px;
                height: 50px;
                border-radius: 8px;
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 24px;
                color: white;
                margin-right: 15px;
            }
            .icon-blue {
                background-color: #0a73e8;
            }
            .icon-red {
                background-color: #ff4b5c;
            }
            .icon-yellow {
                background-color: #ffc107;
            }
            .contact-item-text h4 {
                margin: 0 0 5px 0 !important;
                font-size: 18px;
                font-weight: bold;
                color: #333;
            }
            .contact-item-text p {
                margin: 0 !important;
                font-size: 14px;
                color: #666;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <!-- Load Google Material Icons -->
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    
    <div class="contact-left">
        <div class="contact-item">
            <div class="contact-item-icon icon-blue">
                <span class="material-icons">location_on</span> <!-- Google Material Icon -->
            </div>
            <div class="contact-item-text">
                <h4>Our Location</h4>
                <p> AUB,Beirut,Lebanon</p>
            </div>
        </div>
        <div class="contact-item">
            <div class="contact-item-icon icon-red">
                <span class="material-icons">phone</span> <!-- Google Material Icon -->
            </div>
            <div class="contact-item-text">
                <h4>Call Us</h4>
                <p>+961 03 456 789</p>
            </div>
        </div>
        <div class="contact-item">
            <div class="contact-item-icon icon-yellow">
                <span class="material-icons">email</span> <!-- Google Material Icon -->
            </div>
            <div class="contact-item-text">
                <h4>Email Us</h4>
                <p>chatcourse@gmail.com</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    inject_css2()

def about_page():
    inject_css()

    bg_image_path = r"img\page-header.jpg"
    torn_edge_path = r"img\overlay-top.png"

    bg_image_url = set_overlay_bg_image(bg_image_path)
    torn_edge_url = set_overlay_bg_image(torn_edge_path)

    st.markdown(f"""
        <style>
            .overlay-container {{
                position: relative;
                width: 100%;
                height: 300px;
                background-image: url("{bg_image_url}");
                background-size: cover;
                background-position: center;
                display: flex;
                justify-content: center;
                align-items: center;
                overflow: hidden;
            }}
            .overlay {{
                background-color: rgba(58, 118, 240, 0.6);
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 1;
            }}
            .overlay-content {{
                position: relative;
                z-index: 2;
                color: white;
                text-align: center;
                font-family: Arial, sans-serif;
            }}
            .overlay-content h1 {{
                font-size: 36px;
                font-weight: bold;
            }}
            .overlay-content h2 {{
                font-size: 18px;
                margin-top: 10px;
            }}
            .torn-edge {{
                position: absolute;
                bottom: -17px;
                left: 0;
                width: 100%;
                height: 60px;
                background-image: url("{torn_edge_url}");
                background-size: cover;
                background-repeat: no-repeat;
                transform: scaleY(-1);
                z-index: 2;
            }}
            .section-wrapper {{
            margin-top: 50px; 
            }}
            .about-title {{
                text-align: left; 
                width: 100%; 
                font-size: 24px;
                font-weight: normal; 
                margin: 0 0 10px 0; 
                padding: 0; 
            }}
            .about-section {{
                display: flex;
                flex-direction: row-reverse; 
                align-items: flex-start;
                justify-content: space-between;
                color: white; 
                padding: 20px;
                border-radius: 10px; 
            }}
            .about-content {{
                flex: 1;
                max-width: 700px; 
                margin: 10px;
                line-height: 1.6;
                font-size: 16px;
            }}
            .about-content h2 {{
                color: #1C1C44;
                margin-bottom: 20px;
                text-align: left;
                margin-left: -10px; 
            }}
            .about-content p {{
                font-size: 16px;
                line-height: 1.8;
                color: #444;
            }}
            .about-content {{
                flex: 1;
                max-width: 50%; 
                margin: 0; 
                padding-top: 0; 
            }}
            .about-image {{
                position: relative;
                top: 180px;
                flex: 1;
                max-width: 50%; 
                margin-right: 20px; 
                margin-left: 20px; 
                display: flex;
                justify-content: flex-start; 
                align-items: flex-start; 
            }}
            .about-image img{{
                width: 100%; 
                height: auto; 
            }}
            .stats-section {{
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
                gap: 0px; 
            }}
            .stat-card {{
                flex: 0 1 25px; 
                text-align: center;
                padding: 20px;
                border-radius: 0;
                color: white;
                font-family: Arial, sans-serif;
            }}
            .stat-card .stat-number {{
                font-size: 36px;
                font-weight: bold;
            }}
            .stat-card .stat-label {{
                font-size: 16px;
                margin-top: 10px;
            }}
            .stat-card.green {{
                background-color: #28a745;
            }}

            .stat-card.blue {{
                background-color: #007bff; 
            }}

            .stat-card.red {{
                background-color: #dc3545; 
            }}
            .stat-card.yellow {{
                background-color: #ffc107; 
            }}
            .stat-box {{
                text-align: center;
                padding: 20px;
                border-radius: 10px;
                color: white;
                width: 150px;
                font-family: Arial, sans-serif;
            }}
            .green {{ background-color: #28A745; }}
            .blue {{background-color: #007BFF; }}
            .red {{background-color: #DC3545; }}
            .yellow {{background-color: #FFC107; }}
            .feature-block {{
                display: flex;
                margin: 20px 0;
                align-items: center;
            }}
            .feature-block img {{
                background-color: #004CFF;
                padding: 20px;
                border-radius: 50%;
                width: 70px;
                height: 70px;
                margin-right: 20px;
            }}
            .features-section {{
                display: flex;
                flex-direction: column;
                gap: 20px;
                padding: 10px;
                margin-bottom: 40px; 
            }}
            .feature-item {{
                display: flex;
                align-items: center;
                gap: 20px;
                padding: 15px;
                border-radius: 8px;
                background-color: #ffffff; 
                box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1); 
                max-width: 600px; 
                overflow: hidden; 
            }}
            .feature-icon {{
                display: flex;
                justify-content: center;
                align-items: center;
                width: 50px;
                height: 70px;
                border-radius: 10px;
            }}
            .feature-icon-blue {{
                background-color: #007bff; 
            }}
            .feature-icon-red {{
                background-color: #dc3545;
            }}
            .feature-icon-yellow {{
                background-color: #ffc107; 
            }}
            .feature-icon img {{
                width: 40px;
                height: 40px;
            }}
            .feature-text h3 {{
                margin: 0 0 5px 0;
                font-size: 18px;
                font-weight: bold;
                color: #1c1c44;
            }}
            .feature-text p {{
                margin: 0;
                font-size: 14px;
                line-height: 1.6;
                color: #555; 
            }}
            .graduate-image {{
                position: relative;
                top: 50px;
                flex: 1;
                max-width: 400px; 
                max-height: 400px;
                margin-right: 20px; 
                margin-left: 20px; 
                display: flex;
                justify-content: flex-start; 
                align-items: flex-start; 
            }}
            .graduate-image img{{
                width: 100%; 
                height: auto; 
            }}
        </style>

    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="overlay-container">
            <div class="overlay"></div>
            <div class="overlay-content">
                <h1>About</h1>
                <h2>Home &raquo; About</h2>
            </div>
        </div>
        <div class="torn-edge"></div>
    """, unsafe_allow_html=True)

    image_path = r"img\about.jpg"
    about_image = Image.open(image_path)
    base64_image = encode_image_to_base64(about_image)

    st.markdown(f"""
        <div class="about-section">
            <div class="about-image">
                <img src="data:image/png;base64,{base64_image}" alt="Graduate style="position: absolute; top: -30px;">
            </div>
            <div class="about-content">
                <p style="color: red; font-size: 18px; font-weight: bold; text-align: left; margin-bottom: 5px;">ABOUT US</p>
                <h2 style="text-align: left; font-size: 30px; font-weight: bold; margin-bottom: 20px;">First Choice For Online Education Anywhere</h2>
                <p>Experience a revolutionary way to learn with our AI-powered educational platform. Combining the latest in artificial intelligence technology, our platform offers personalized learning experiences through an interactive AI chatbot, engaging flashcards, and adaptive learning tools. Whether you're preparing for exams, mastering new skills, or expanding your knowledge, our platform tailors content to your unique needs, helping you learn faster and more effectively. Discover a smarter, more efficient way to achieve your educational goals with the power of AI at your fingertips.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="stats-section">
            <div class="stat-card green">
                <div class="stat-label">AVAILABLE SUBJECTS</div>
            </div>
            <div class="stat-card blue">
                <div class="stat-label">ONLINE COURSES</div>
            </div>
            <div class="stat-card red">
                <div class="stat-label">SKILLED INSTRUCTORS</div>
            </div>
            <div class="stat-card yellow">
                <div class="stat-label">HAPPY STUDENTS</div>
            </div>
        </div>

    """, unsafe_allow_html=True)

    image_path = r"img\feature.jpg"
    about_image = Image.open(image_path)
    base64_img = encode_image_to_base64(about_image)
    st.markdown(f"""
        <div class="about-section">
            <div class="graduate-image">
                <img class="graduate-image" src="data:image/png;base64,{base64_img}" alt="Graduate ">
            </div>
            <div class="about-content">
                <p style="color: red; font-size: 18px; font-weight: bold; text-align: left; margin-bottom: 5px;">WHY CHOOSE US</p>
                <h2 style="text-align: left; font-size: 30px; font-weight: bold; margin-bottom: 20px;">Why You Should Start Learning with Us?</h2>
                <p>Join a platform that truly understands your learning needs. Our commitment to innovation ensures a unique, engaging, and personalized education experience. With tools designed to simplify complex topics and foster deep understanding, we empower learners to achieve their goals faster and more effectively. Experience the perfect blend of technology and expertise to take your learning journey to the next level. Let’s make your success our mission!</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="features-section">
        <div class="feature-item">
            <div class="feature-icon feature-icon-blue">
                <img src="https://img.icons8.com/ios-filled/50/ffffff/graduation-cap.png" alt="Skilled Instructors Icon">
            </div>
            <div class="feature-text">
                <h3>Skilled Instructors</h3>
                <p>Learn from top professionals with years of teaching and industry experience. Our instructors are here to guide you every step of the way.</p>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon feature-icon-red">
                <img src="https://img.icons8.com/ios-filled/50/ffffff/certificate.png" alt="International Certificate Icon">
            </div>
            <div class="feature-text">
                <h3>International Certificate</h3>
                <p>Earn certificates recognized globally, enhancing your credentials and opening up opportunities worldwide.</p>
            </div>
        </div>
        <div class="feature-item">
            <div class="feature-icon feature-icon-yellow">
                <img src="https://img.icons8.com/ios-filled/50/ffffff/class.png" alt="Online Classes Icon">
            </div>
            <div class="feature-text">
                <h3>Online Classes</h3>
                <p>Access flexible and engaging online classes designed to fit into your busy schedule, anytime, anywhere.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    inject_css2()