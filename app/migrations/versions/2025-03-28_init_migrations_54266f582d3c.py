# type: ignore
"""Init migrations

Revision ID: 54266f582d3c
Revises: 
Create Date: 2025-03-28 00:38:36.733896

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
revision = '54266f582d3c'
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
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_addresses')),
    sa.UniqueConstraint('id'),
    sa.UniqueConstraint('id', name=op.f('uq_addresses_id'))
    )
    op.create_table('images',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('url', sa.String(), nullable=False),
    sa.Column('model_id', sa.UUID(), nullable=True),
    sa.Column('model_type', sa.String(length=50), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_images')),
    sa.UniqueConstraint('id'),
    sa.UniqueConstraint('id', name=op.f('uq_images_id'))
    )
    op.create_table('roles',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(length=50), nullable=False),
    sa.Column('slug', sa.String(length=100), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_roles')),
    sa.UniqueConstraint('id'),
    sa.UniqueConstraint('id', name=op.f('uq_roles_id')),
    sa.UniqueConstraint('name'),
    sa.UniqueConstraint('name', name=op.f('uq_roles_name')),
    sa.UniqueConstraint('slug', name='uq_roles_slug')
    )
    with op.batch_alter_table('roles', schema=None) as batch_op:
        batch_op.create_index('ix_roles_slug_unique', ['slug'], unique=True)

    op.create_table('tags',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('type', sa.String(length=255), nullable=True),
    sa.Column('slug', sa.String(length=100), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_tags')),
    sa.UniqueConstraint('id'),
    sa.UniqueConstraint('id', name=op.f('uq_tags_id')),
    sa.UniqueConstraint('name'),
    sa.UniqueConstraint('name', name=op.f('uq_tags_name')),
    sa.UniqueConstraint('slug', name='uq_tags_slug')
    )
    with op.batch_alter_table('tags', schema=None) as batch_op:
        batch_op.create_index('ix_tags_slug_unique', ['slug'], unique=True)

    op.create_table('image_tags',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('image_id', sa.UUID(), nullable=False),
    sa.Column('tag_id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['image_id'], ['images.id'], name=op.f('fk_image_tags_image_id_images'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['tag_id'], ['tags.id'], name=op.f('fk_image_tags_tag_id_tags'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_image_tags')),
    sa.UniqueConstraint('id'),
    sa.UniqueConstraint('id', name=op.f('uq_image_tags_id'))
    )
    op.create_table('users',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('name', sa.String(length=255), nullable=False),
    sa.Column('email', sa.String(length=255), nullable=False),
    sa.Column('phone', sa.String(length=20), nullable=True),
    sa.Column('password', sa.Text(), nullable=False),
    sa.Column('verified', sa.Boolean(), nullable=False),
    sa.Column('address_id', sa.UUID(), nullable=True),
    sa.Column('image_id', sa.UUID(), nullable=True),
    sa.Column('device_token', sa.String(length=255), nullable=True),
    sa.Column('reset_password_token', sa.String(length=64), nullable=True),
    sa.Column('reset_password_expires', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['address_id'], ['addresses.id'], name=op.f('fk_users_address_id_addresses'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['image_id'], ['images.id'], name=op.f('fk_users_image_id_images'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_users')),
    sa.UniqueConstraint('address_id'),
    sa.UniqueConstraint('address_id', name=op.f('uq_users_address_id')),
    sa.UniqueConstraint('device_token'),
    sa.UniqueConstraint('device_token', name=op.f('uq_users_device_token')),
    sa.UniqueConstraint('email'),
    sa.UniqueConstraint('email', name=op.f('uq_users_email')),
    sa.UniqueConstraint('id'),
    sa.UniqueConstraint('id', name=op.f('uq_users_id')),
    sa.UniqueConstraint('image_id'),
    sa.UniqueConstraint('image_id', name=op.f('uq_users_image_id'))
    )
    op.create_table('partner_registrations',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('type', sa.Enum('INDIVIDUAL', 'ENTERPRISE', name='partnertype'), nullable=False),
    sa.Column('date_of_birth', sa.Date(), nullable=True),
    sa.Column('tax_id', sa.String(length=100), nullable=True),
    sa.Column('authorized_representative_name', sa.String(length=255), nullable=True),
    sa.Column('approved', sa.Boolean(), nullable=True),
    sa.Column('reject_reason', sa.String(length=255), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
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
    sa.Column('bathrooms', sa.Integer(), nullable=False),
    sa.Column('sqm', sa.Numeric(precision=6, scale=2), nullable=True),
    sa.Column('status', sa.Boolean(), server_default='false', nullable=False),
    sa.Column('active', sa.Boolean(), server_default='false', nullable=False),
    sa.Column('owner_id', sa.UUID(), nullable=True),
    sa.Column('address_id', sa.UUID(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
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
    op.create_table('user_tags',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('tag_id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['tag_id'], ['tags.id'], name=op.f('fk_user_tags_tag_id_tags')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_user_tags_user_id_users')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_user_tags')),
    sa.UniqueConstraint('id'),
    sa.UniqueConstraint('id', name=op.f('uq_user_tags_id'))
    )
    op.create_table('favorites',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.UUID(), nullable=False),
    sa.Column('property_id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['property_id'], ['properties.id'], name=op.f('fk_favorites_property_id_properties'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_favorites_user_id_users'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_favorites')),
    sa.UniqueConstraint('id'),
    sa.UniqueConstraint('id', name=op.f('uq_favorites_id'))
    )
    op.create_table('property_tags',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('property_id', sa.UUID(), nullable=False),
    sa.Column('tag_id', sa.UUID(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['property_id'], ['properties.id'], name=op.f('fk_property_tags_property_id_properties'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['tag_id'], ['tags.id'], name=op.f('fk_property_tags_tag_id_tags'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_property_tags')),
    sa.UniqueConstraint('id'),
    sa.UniqueConstraint('id', name=op.f('uq_property_tags_id'))
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
    op.drop_table('property_tags')
    op.drop_table('favorites')
    op.drop_table('user_tags')
    op.drop_table('user_roles')
    op.drop_table('properties')
    op.drop_table('partner_registrations')
    op.drop_table('users')
    op.drop_table('image_tags')
    with op.batch_alter_table('tags', schema=None) as batch_op:
        batch_op.drop_index('ix_tags_slug_unique')

    op.drop_table('tags')
    with op.batch_alter_table('roles', schema=None) as batch_op:
        batch_op.drop_index('ix_roles_slug_unique')

    op.drop_table('roles')
    op.drop_table('images')
    with op.batch_alter_table('addresses', schema=None) as batch_op:
        batch_op.drop_index('idx_addresses_coordinates', postgresql_using='gist')

    op.drop_table('addresses')
    # ### end Alembic commands ###

def data_upgrades() -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    roles_table = sa.table(
        "roles",
        sa.column("id", sa.UUID()),
        sa.column("name", sa.String(length=50)),
        sa.column("slug", sa.String(length=100)),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )
    import uuid

    op.bulk_insert(
        roles_table,
        [
            {
                "id": uuid.UUID("550e8400-e29b-41d4-a716-446655440000"),
                "name": "admin",
                "slug": "admin",
                "created_at": now,
                "updated_at": now,
            },
            {
                "id": uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479"),
                "name": "partner",
                "slug": "partner",
                "created_at": now,
                "updated_at": now,
            },
            {
                "id": uuid.UUID("9d3c5f77-a6a3-4f3e-8c21-3d9a5e6b7f2c"),
                "name": "customer",
                "slug": "customer",
                "created_at": now,
                "updated_at": now,
            },
        ],
    )

def data_downgrades() -> None:
    """Add any optional data downgrade migrations here!"""
