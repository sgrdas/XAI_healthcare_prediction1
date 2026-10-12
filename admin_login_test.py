import os

from app import create_app
from werkzeug.datastructures import MultiDict

email = os.environ.get('TEST_ADMIN_EMAIL')
password = os.environ.get('TEST_ADMIN_PASSWORD')
if not email or not password:
    raise SystemExit('Set TEST_ADMIN_EMAIL and TEST_ADMIN_PASSWORD before running this test.')

app = create_app('development')
app.config['TESTING'] = True
app.config['WTF_CSRF_ENABLED'] = False

with app.test_client() as client:
    response = client.get('/auth/login')
    print('GET /auth/login', response.status_code)

    data = MultiDict([
        ('email', email),
        ('password', password),
        ('remember_me', 'y'),
    ])
    response = client.post('/auth/login', data=data, follow_redirects=False)
    print('POST /auth/login', response.status_code)
    print('Location:', response.headers.get('Location'))
    print('Body snippet:')
    print(response.get_data(as_text=True)[:1000])

    if response.status_code == 302:
        follow = client.get(response.headers.get('Location'))
        print('Follow redirect', follow.status_code)
        print('Redirect body snippet:')
        print(follow.get_data(as_text=True)[:1000])
