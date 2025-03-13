# type: ignore
"""Initial Migration

Revision ID: 108fec78e778
Revises: 
Create Date: 2025-03-13 15:33:33.212907

"""

import warnings
from typing import TYPE_CHECKING
import datetime
import sqlalchemy as sa
from alembic import op
from advanced_alchemy.types import EncryptedString, EncryptedText, GUID, ORA_JSONB, DateTimeUTC
from sqlalchemy import Text  # noqa: F401
from geoalchemy2 import Geography

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["downgrade", "upgrade", "schema_upgrades", "schema_downgrades", "data_upgrades", "data_downgrades"]

sa.GUID = GUID
sa.DateTimeUTC = DateTimeUTC
sa.ORA_JSONB = ORA_JSONB
sa.EncryptedString = EncryptedString
sa.EncryptedText = EncryptedText

# revision identifiers, used by Alembic.
revision = '108fec78e778'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with op.get_context().autocommit_block():
            schema_upgrades()
            data_upgrades()

def downgrade() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        with op.get_context().autocommit_block():
            data_downgrades()
            schema_downgrades()

def schema_upgrades() -> None:
    """schema upgrade migrations go here."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('addresses',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('street', sa.String(length=255), nullable=False),
    sa.Column('city', sa.String(length=100), nullable=False),
    sa.Column('postal_code', sa.String(length=20), nullable=True),
    sa.Column('neighborhood', sa.String(length=100), nullable=True),
    sa.Column('latitude', sa.Numeric(precision=12, scale=9), nullable=False),
    sa.Column('longitude', sa.Numeric(precision=12, scale=9), nullable=False),
    sa.Column('coordinates', Geography(geometry_type='POINT', srid=4326, from_text='ST_GeogFromText', name='geography'), nullable=True),
    sa.Column('geohash', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_addresses')),
    sa.UniqueConstraint('id'),
    sa.UniqueConstraint('id', name=op.f('uq_addresses_id'))
    )
    op.create_table('roles',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(length=50), nullable=False),
    sa.Column('slug', sa.String(length=100), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_roles')),
    sa.UniqueConstraint('id'),
    sa.UniqueConstraint('id', name=op.f('uq_roles_id')),
    sa.UniqueConstraint('name'),
    sa.UniqueConstraint('name', name=op.f('uq_roles_name')),
    sa.UniqueConstraint('slug', name='uq_roles_slug')
    )
    with op.batch_alter_table('roles', schema=None) as batch_op:
        batch_op.create_index('ix_roles_slug_unique', ['slug'], unique=True)

    op.create_table('users',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('email', sa.String(length=255), nullable=False),
    sa.Column('phone', sa.String(length=20), nullable=True),
    sa.Column('password', sa.Text(), nullable=False),
    sa.Column('verified', sa.Boolean(), nullable=False),
    sa.Column('address_id', sa.UUID(), nullable=True),
    sa.Column('device_token', sa.String(length=255), server_default='NULL', nullable=True),
    sa.Column('reset_password_token', sa.String(length=64), nullable=True),
    sa.Column('reset_password_expires', sa.DateTime(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['address_id'], ['addresses.id'], name=op.f('fk_users_address_id_addresses'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_users')),
    sa.UniqueConstraint('address_id'),
    sa.UniqueConstraint('address_id', name=op.f('uq_users_address_id')),
    sa.UniqueConstraint('device_token'),
    sa.UniqueConstraint('device_token', name=op.f('uq_users_device_token')),
    sa.UniqueConstraint('email'),
    sa.UniqueConstraint('email', name=op.f('uq_users_email')),
    sa.UniqueConstraint('id'),
    sa.UniqueConstraint('id', name=op.f('uq_users_id'))
    )
    op.create_table('partner_registrations',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('type', sa.Enum('INDIVIDUAL', 'ENTERPRISE', name='partner_type'), nullable=False),
    sa.Column('profile_url', sa.String(length=255), nullable=True),
    sa.Column('date_of_birth', sa.Date(), nullable=True),
    sa.Column('business_registration_certificate_url', sa.String(length=255), nullable=True),
    sa.Column('tax_id', sa.String(length=100), nullable=True),
    sa.Column('authorized_representative_name', sa.String(length=255), nullable=True),
    sa.Column('approved', sa.String(length=100), server_default='NULL', nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_partner_registrations_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_partner_registrations')),
    sa.UniqueConstraint('id'),
    sa.UniqueConstraint('id', name=op.f('uq_partner_registrations_id')),
    sa.UniqueConstraint('user_id'),
    sa.UniqueConstraint('user_id', name=op.f('uq_partner_registrations_user_id'))
    )
    op.create_table('properties',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('title', sa.String(length=255), nullable=False),
    sa.Column('property_category', sa.String(length=50), nullable=False),
    sa.Column('transaction_type', sa.String(length=50), nullable=False),
    sa.Column('price', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('bedrooms', sa.Integer(), nullable=False),
    sa.Column('bathrooms', sa.Numeric(precision=3, scale=1), nullable=False),
    sa.Column('sqm', sa.Integer(), nullable=True),
    sa.Column('status', sa.String(length=50), nullable=False),
    sa.Column('active', sa.Boolean(), server_default='false', nullable=False),
    sa.Column('owner_id', sa.UUID(), nullable=True),
    sa.Column('address_id', sa.UUID(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['address_id'], ['addresses.id'], name=op.f('fk_properties_address_id_addresses'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], name=op.f('fk_properties_owner_id_users'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_properties')),
    sa.UniqueConstraint('address_id'),
    sa.UniqueConstraint('address_id', name=op.f('uq_properties_address_id')),
    sa.UniqueConstraint('id'),
    sa.UniqueConstraint('id', name=op.f('uq_properties_id'))
    )
    op.create_table('user_roles',
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('role_id', sa.UUID(), nullable=False),
    sa.ForeignKeyConstraint(['role_id'], ['roles.id'], name=op.f('fk_user_roles_role_id_roles')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_user_roles_user_id_users')),
    sa.PrimaryKeyConstraint('user_id', 'role_id', name=op.f('pk_user_roles'))
    )
    # ### end Alembic commands ###

def schema_downgrades() -> None:
    """schema downgrade migrations go here."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('spatial_ref_sys',
    sa.Column('srid', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('auth_name', sa.VARCHAR(length=256), autoincrement=False, nullable=True),
    sa.Column('auth_srid', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('srtext', sa.VARCHAR(length=2048), autoincrement=False, nullable=True),
    sa.Column('proj4text', sa.VARCHAR(length=2048), autoincrement=False, nullable=True),
    sa.CheckConstraint('srid > 0 AND srid <= 998999', name='spatial_ref_sys_srid_check'),
    sa.PrimaryKeyConstraint('srid', name='spatial_ref_sys_pkey')
    )
    op.drop_table('user_roles')
    op.drop_table('properties')
    op.drop_table('partner_registrations')
    op.drop_table('users')
    with op.batch_alter_table('roles', schema=None) as batch_op:
        batch_op.drop_index('ix_roles_slug_unique')

    op.drop_table('roles')
    with op.batch_alter_table('addresses', schema=None) as batch_op:
        batch_op.drop_index('idx_addresses_coordinates', postgresql_using='gist')

    op.drop_table('addresses')
    # ### end Alembic commands ###

def data_upgrades() -> None:
    # Get the current UTC time to set as created_at and updated_at
    now = datetime.datetime.utcnow()
    # Define a temporary table for bulk insert into the "roles" table.
    roles_table = sa.table(
        'roles',
        sa.column('id', sa.String(length=36)),
        sa.column('name', sa.String(length=50)),
        sa.column('slug', sa.String(length=100)),
        sa.column('created_at', sa.DateTime),
        sa.column('updated_at', sa.DateTime),
    )
    import uuid
    op.bulk_insert(roles_table, [
        {
            'id': uuid.UUID('11111111-1111-1111-1111-111111111111'),
            'name': 'admin',
            'slug': 'admin',
            'created_at': now,
            'updated_at': now,
        },
        {
            'id': uuid.UUID('22222222-2222-2222-2222-222222222222'),
            'name': 'partner',
            'slug': 'partner',
            'created_at': now,
            'updated_at': now,
        },
        {
            'id': uuid.UUID('33333333-3333-3333-3333-333333333333'),
            'name': 'customer',
            'slug': 'customer',
            'created_at': now,
            'updated_at': now,
        },
    ])

def data_downgrades() -> None:
    # Remove the roles based on their unique slug values.
    op.execute(
        "DELETE FROM roles WHERE slug IN ('admin', 'partner', 'customer')"
    )